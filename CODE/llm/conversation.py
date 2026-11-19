"""One conversational turn end to end.

    user message
      -> extract_turn (LLM, structured output)        constraints.py
      -> DialogueState.apply_delta (conflict rules)   dialogue_state.py
      -> next_question? -> ask ONE question           clarification.py
      -> build_query_text -> retriever.search(top_n)  retrieval/semantic_search.py
      -> ranker(retrieved, constraints, products)     Disha's stage (stub for now)
      -> explain_recommendations (RAG)                explanation.py

Every turn returns a dict (see TurnResponse keys in CODE/README.md) so a
backend or evaluation script can consume it directly.
"""

from llm import stub_ranker
from llm.clarification import next_question
from llm.constraints import build_query_text, extract_turn
from llm.dialogue_state import DialogueState
from llm.explanation import explain_recommendations


class GiftConcierge:
    def __init__(self, llm, retriever, products, ranker=stub_ranker.rank, kg_facts=None,
                 top_n=100, top_k=5, log_path=None):
        self.llm = llm
        self.retriever = retriever
        self.products = products
        self.ranker = ranker
        self.kg_facts = kg_facts
        self.top_n = top_n
        self.top_k = top_k
        self.state = DialogueState(log_path=log_path)
        self.last_assistant = None

    def _reply(self, kind, message, **extra):
        self.last_assistant = message
        return {"type": kind, "message": message, **extra}

    def handle(self, user_message):
        extraction = extract_turn(self.llm, user_message, self.state.constraints,
                                  self.last_assistant)
        update = self.state.apply_delta(extraction.delta, user_message)
        base = {
            "turn": update.turn,
            "constraints": self.state.constraints.to_handoff(),
            "state_changes": [vars(c) for c in update.changes],
            "conflicts": [vars(c) for c in update.conflicts],
            "extraction_error": extraction.error,
        }

        if extraction.error:
            return self._reply("question",
                               "Sorry, I didn't catch that. Could you rephrase who the gift "
                               "is for, what they like, and your budget?",
                               slot="rephrase", **base)

        question = next_question(self.state, update)
        if question is not None:
            return self._reply("question", question.question, slot=question.slot, **base)

        query_text = build_query_text(self.state.constraints)
        retrieved = self.retriever.search(query_text, top_n=self.top_n)
        ranked = self.ranker(retrieved, base["constraints"], self.products, top_k=self.top_k)
        if not ranked:
            return self._reply(
                "question",
                "I couldn't find anything that fits all of those constraints. Could you "
                "raise the budget or loosen one of the requirements?",
                slot="no_results", query_text=query_text, retrieved=retrieved,
                ranked=ranked, **base)

        explained = explain_recommendations(self.llm, ranked, self.products,
                                            self.state.constraints, self.kg_facts, self.top_k)
        lines = ["Here are some ideas:"]
        for item in explained["explanations"]:
            product = self.products[item["parent_asin"]]
            lines.append(f"{item['rank']}. {product['title']} (${product['price_num']:,.2f})\n"
                         f"   {item['explanation']}")
        lines.append("Want something cheaper, different, or more like one of these?")
        return self._reply("recommendations", "\n".join(lines), query_text=query_text,
                           retrieved=retrieved, ranked=ranked,
                           explanations=explained["explanations"],
                           evidence=explained["evidence"], **base)
