"""Knowledge-graph evidence for explanations (contract kg-evidence/1.x).

    from knowledge_graph.api import load_kg
    GiftConcierge(..., kg_facts=KGEvidence(load_kg()))

Only grounding.assertable_claims and grounding.cautions become facts. Single
review observations are not used, because the contract says not to generalize
them. found=false products get no facts.
"""

import logging

from llm.explanation import MAX_KG_FACTS

log = logging.getLogger(__name__)

MAX_CAUTIONS = 2


def facts_from_result(result, max_facts=MAX_KG_FACTS):
    """Prompt facts for one get_evidence() result, strongest first.

    Order: interests matching the request, review aggregates, the rest. Room is
    reserved for cautions so praise can never crowd out a known concern.
    """
    if not result.get("found"):
        return []
    grounding = result["grounding"]
    matched = {i["path_id"] for i in result["interests"] if i["matches_query"]}

    def priority(claim):
        if claim["path_id"] in matched:
            return 0
        return 1 if claim["basis"] == "review_aggregate" else 2

    cautions = [{"text": f"Caution: {c['text']}", "path_id": c["path_id"]}
                for c in grounding["cautions"][:MAX_CAUTIONS]]
    claims = [{"text": c["text"], "path_id": c["path_id"]}
              for c in sorted(grounding["assertable_claims"], key=priority)]
    return claims[:max_facts - len(cautions)] + cautions


class KGEvidence:
    def __init__(self, kg, max_reviews=3):
        self.kg = kg
        self.max_reviews = max_reviews

    def facts_for(self, parent_asins, interest_labels=()):
        """{parent_asin: [{"text", "path_id"}]} for the ranked products."""
        response = self.kg.get_evidence(list(parent_asins),
                                        query_interests=list(interest_labels) or None,
                                        max_reviews=self.max_reviews)
        for w in response["warnings"]:
            log.warning("KG %s: %s", w["code"], w.get("value"))
        return {r["parent_asin"]: facts_from_result(r) for r in response["results"]}

    def get_facts(self, parent_asin):
        return self.facts_for([parent_asin]).get(parent_asin, [])
