"""End-to-end turn handling with a fake LLM and fake retriever (no API, no model)."""

from llm.constraints import TURN_DELTA_SCHEMA
from llm.conversation import GiftConcierge
from llm.offline_llm import extract
from llm.stub_ranker import passes_filters, rank

PRODUCTS = {
    "T1": {"parent_asin": "T1", "title": "Dinosaur Puzzle", "domain": "Toys & Games",
           "categories": ["Toys & Games", "Puzzles"], "price_num": 20.0,
           "average_rating": 4.5, "age_min": 5, "age_max": 10},
    "T2": {"parent_asin": "T2", "title": "Dinosaur Plush", "domain": "Toys & Games",
           "categories": ["Toys & Games", "Stuffed Animals"], "price_num": 45.0,
           "average_rating": 4.9, "age_min": None, "age_max": None},
    "B1": {"parent_asin": "B1", "title": "Dinosaur Encyclopedia", "domain": "Books",
           "categories": ["Books", "Science"], "price_num": 12.0,
           "average_rating": 4.0, "age_min": 6, "age_max": 12},
    "T3": {"parent_asin": "T3", "title": "Baby Rattle", "domain": "Toys & Games",
           "categories": ["Toys & Games", "Baby"], "price_num": 9.0,
           "average_rating": 5.0, "age_min": 0, "age_max": 1},
}


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query_text, top_n=100):
        self.queries.append(query_text)
        order = ["T1", "T2", "B1", "T3"]
        return [{"parent_asin": a, "score": 0.9 - 0.1 * i, "rank": i + 1}
                for i, a in enumerate(order[:top_n])]


class ScriptedLLM:
    """Extraction via the offline rule extractor; explanations cite the title."""

    def complete_json(self, messages, schema, name, system=None):
        if name == "gift_constraints_delta":
            return extract(messages[-1]["content"].split("Latest user message:\n", 1)[-1])
        pids = [line.split(":")[0].replace("Product ", "")
                for line in messages[-1]["content"].splitlines() if line.startswith("Product P")]
        return {"explanations": [{"product_id": p, "explanation": "Fits.",
                                  "evidence_ids": [f"{p}.title"]} for p in pids]}

    def complete(self, *a, **k):
        return ""


# ---------- stub ranker ----------

def test_stub_ranker_filters_and_orders():
    retrieved = FakeRetriever().search("x")
    constraints = {"budget_min": None, "budget_max": 30, "exclusions": ["puzzles"],
                   "category_preference": None, "target_age_range": [7, 7]}
    out = rank(retrieved, constraints, PRODUCTS, top_k=5)
    # T1 excluded (puzzle), T2 too expensive, T3 wrong age -> only the book
    assert out == [{"parent_asin": "B1", "score": 0.72, "rank": 1}]


def test_stub_ranker_category_and_missing_ages():
    c = {"category_preference": "Toys & Games", "exclusions": [], "target_age_range": [4, 12]}
    assert passes_filters(PRODUCTS["T2"], c)          # unknown age range is allowed
    assert not passes_filters(PRODUCTS["B1"], c)      # wrong domain
    assert not passes_filters(PRODUCTS["T3"], c)      # baby toy


# ---------- conversation ----------

def test_full_conversation_flow():
    retriever = FakeRetriever()
    bot = GiftConcierge(ScriptedLLM(), retriever, PRODUCTS, top_k=3)

    r1 = bot.handle("gift for my 7 year old nephew")
    assert r1["type"] == "question" and r1["slot"] == "interests"
    assert retriever.queries == []                     # no retrieval before clarifying

    r2 = bot.handle("he loves dinosaurs")
    assert r2["slot"] == "budget"

    r3 = bot.handle("under $50")
    assert r3["type"] == "recommendations"
    assert r3["query_text"] == "gift for my 7 year old nephew, age 7 who likes dinosaurs"
    assert [x["parent_asin"] for x in r3["ranked"]] == ["T1", "T2", "B1"]
    assert r3["explanations"][0]["evidence_ids"] == ["P1.title"]
    assert r3["constraints"]["budget_max"] == 50

    r4 = bot.handle("cheaper")
    assert r4["constraints"]["budget_max"] == 37.5
    assert [c["rule"] for c in r4["state_changes"]] == ["relative_cheaper"]
    assert "T2" not in [x["parent_asin"] for x in r4["ranked"]]

    r5 = bot.handle("at least $40 but under $20")
    assert r5["type"] == "question" and r5["slot"] == "conflict:budget_min_gt_max"
    assert r5["constraints"]["budget_max"] == 37.5    # unchanged until the user decides


def test_extraction_failure_asks_to_rephrase():
    class BrokenLLM(ScriptedLLM):
        def complete_json(self, messages, schema, name, system=None):
            raise RuntimeError("503")
    bot = GiftConcierge(BrokenLLM(), FakeRetriever(), PRODUCTS)
    r = bot.handle("hello")
    assert r["slot"] == "rephrase" and "503" in r["extraction_error"]


def test_no_results_asks_to_loosen():
    bot = GiftConcierge(ScriptedLLM(), FakeRetriever(), PRODUCTS)
    bot.handle("for my dad, he likes dinosaurs, under $5")
    r = bot.handle("under $5")
    assert r["slot"] == "no_results" and r["ranked"] == []


def test_offline_extractor_returns_schema_complete_delta():
    d = extract("for my mom, she loves gardening and tea, around $50, no candles")
    assert set(d) == set(TURN_DELTA_SCHEMA["properties"])
    assert d["recipient"] == "my mom"
    assert d["interests_add"] == ["gardening", "tea"]
    assert (d["budget_min"], d["budget_max"]) == (40, 60)
    assert d["exclusions_add"] == ["candles"]
