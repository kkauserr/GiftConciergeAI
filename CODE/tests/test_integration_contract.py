"""Shared contract with ranking, KG and frontend. No API calls, no models.

Tests that need a teammate's code are skipped until that code is merged.
"""

import sys
from pathlib import Path

import pytest

from llm.constraints import GiftConstraints, TurnDelta, short_recipient
from llm.conversation import GiftConcierge
from llm.dialogue_state import DialogueState
from llm.exclusions import FILTER_GROUPS, normalize_exclusions
from llm.explanation import MAX_KG_FACTS, collect_evidence
from llm.interest_labels import VOCAB_PATH, interest_labels, label_for, labels_for
from llm.kg_adapter import KGEvidence, facts_from_result
from llm.offline_llm import extract
from llm.ranking_adapter import HybridRanker
from retrieval.export_semantic import export

REPO_ROOT = str(Path(__file__).resolve().parents[2])
needs_vocab = pytest.mark.skipif(not VOCAB_PATH.exists(), reason="KG not merged yet")


# ---------- frontend contract ----------

def test_handoff_matches_agreed_json():
    c = GiftConstraints(recipient="my sister", age=22, occasion="graduation",
                        interests=["coffee", "reading"], budget_max=80, exclusions=["wearable"])
    out = c.to_handoff()
    for key, value in {"recipient": "sister", "recipient_age": 22, "occasion": "graduation",
                       "interests": ["coffee", "reading"], "keywords": [], "budget_min": None,
                       "budget_max": 80, "needs_clarification": False,
                       "clarification_question": None}.items():
        assert out[key] == value
    assert out["exclusions"] == out["exclusions_stated"] == ["wearable"]


def test_question_turn_sets_clarification_fields():
    class LLM:
        def complete_json(self, messages, schema, name, system=None):
            return extract(messages[-1]["content"].split("Latest user message:\n", 1)[-1])
    r = GiftConcierge(LLM(), None, {}).handle("gift for my sister")
    assert r["type"] == "question"
    assert r["constraints"]["needs_clarification"] is True
    assert r["constraints"]["clarification_question"] == r["message"]


@pytest.mark.parametrize("raw,short", [
    ("my sister", "sister"), ("my 8 year old nephew", "nephew"),
    ("a 6-month-old niece", "niece"), ("coworker", "coworker"), (None, None)])
def test_short_recipient(raw, short):
    assert short_recipient(raw) == short


# ---------- exclusions ----------

def test_aliases_and_plain_terms():
    assert normalize_exclusions(["things to wear", "wearable"]) == ["wearable"]
    assert "plush" in normalize_exclusions(["plush toys"])
    assert normalize_exclusions(["mugs"]) == ["mugs"]
    assert normalize_exclusions([]) == []


def test_state_keeps_what_the_user_said():
    state = DialogueState()
    state.apply_delta(TurnDelta(exclusions_add=["wearable"]))
    assert state.constraints.exclusions == ["wearable"]
    state.apply_delta(TurnDelta(interests_add=["wearable"]))     # newer statement wins
    assert state.constraints.to_handoff()["exclusions"] == []


def test_expanded_exclusions_work_in_the_real_filter():
    sys.path.insert(0, REPO_ROOT)
    filters = pytest.importorskip("CODE.ranking.filters")
    check_product = filters.check_product
    # every group we leave to the filter must be one it expands
    assert set(FILTER_GROUPS) <= set(filters.EXCLUSION_EXPANSIONS)
    c = GiftConstraints(exclusions=["things to wear", "plush toys"]).to_handoff()

    def dropped(title, categories, domain="Toys & Games"):
        return bool(check_product({"title": title, "categories": categories, "domain": domain,
                                   "specific_category": categories.split(" > ")[-1]}, c))
    cloth = "Handmade Products > Clothing, Shoes & Accessories > "
    assert dropped("Coffee Costume for Kids", "Toys & Games > Dress Up & Pretend Play > Costumes")
    assert dropped("Realtor V-Neck", cloth + "Men > Clothing > Tops & Tees > T-Shirts")
    assert dropped("Moon Pendant", "Handmade Products > Jewelry > Necklaces")
    assert not dropped("Leather Wallet", cloth + "Luggage & Travel Gear > Wallets")
    assert dropped("Doberman Puppy", "Toys & Games > Stuffed Animals & Plush Toys")
    assert not dropped("Chess Set", "Toys & Games > Board Games")
    assert not dropped("A Shepherd's Watch", "Books > Biographies", "Books")


# ---------- interest labels ----------

@needs_vocab
def test_labels_from_shared_vocabulary():
    assert len(interest_labels()) == 31
    assert label_for("Coffee") == "coffee_tea"
    assert label_for("board games") == "tabletop_games"
    assert label_for("reading") is None
    assert labels_for(["star wars", "coffee"], {"star wars": "scifi_fantasy"}) == \
        ["scifi_fantasy", "coffee_tea"]
    assert labels_for(["ponies"], {"ponies": "not_a_label"}) == []


@needs_vocab
def test_labels_follow_interest_changes():
    state = DialogueState()
    state.apply_delta(TurnDelta.model_validate({
        "interests_add": ["star wars"],
        "interest_labels": [{"interest": "Star Wars", "label": "scifi_fantasy"}]}))
    assert state.constraints.to_handoff()["interest_labels"] == ["scifi_fantasy"]
    state.apply_delta(TurnDelta(interests_remove=["star wars"]))
    assert state.constraints.to_handoff()["interest_labels"] == []


# ---------- KG evidence ----------

def kg_result(n_claims, n_cautions, found=True):
    return {"parent_asin": "A", "found": found,
            "interests": [{"path_id": f"p{n_claims}", "matches_query": True}],
            "grounding": {
                "assertable_claims": [{"text": f"claim {i}", "path_id": f"p{i}",
                                       "basis": "seller_listing"} for i in range(1, n_claims + 1)],
                "cautions": [{"text": f"concern {i}", "path_id": f"c{i}"}
                             for i in range(n_cautions)]}}


def test_kg_facts_put_matches_first_and_keep_cautions():
    facts = facts_from_result(kg_result(8, 1))
    assert len(facts) == MAX_KG_FACTS
    assert facts[0] == {"text": "claim 8", "path_id": "p8"}
    assert facts[-1]["text"] == "Caution: concern 0"
    assert facts_from_result(kg_result(1, 0, found=False)) == []


def test_kg_evidence_carries_path_ids():
    class FakeKG:
        def get_evidence(self, asins, query_interests=None, max_reviews=3):
            self.args = (asins, query_interests)
            return {"warnings": [], "results": [kg_result(1, 0)]}
    kg = FakeKG()
    facts = KGEvidence(kg).facts_for(["A"], ["dogs"])
    assert kg.args == (["A"], ["dogs"])
    ev = collect_evidence({"parent_asin": "A", "title": "T"}, "P1", facts)[-1]
    assert (ev.id, ev.text, ev.path_id) == ("P1.kg[0]", "claim 1", "p1")


def test_real_kg_accepts_our_calls():
    pytest.importorskip("networkx")
    api = pytest.importorskip("knowledge_graph.api")
    facts = KGEvidence(api.load_kg()).facts_for(["B0164RRP14", "NOPE"], ["fishing_hunting"])
    assert facts["NOPE"] == []
    assert facts["B0164RRP14"][0]["text"].startswith("Has a Fishing & Hunting theme")


# ---------- ranking ----------

def test_hybrid_ranker_applies_category_and_age_group():
    def item(asin, domain):
        return {"parent_asin": asin, "domain": domain, "final_score": 0.5,
                "signals": {}, "sources": {}}

    class FakeRanker:
        def recommend(self, constraints, semantic_results=None, config=None):
            self.semantic = semantic_results
            return {"results": [item("B", "Books"), item("OLD", "Toys & Games"),
                                item("OK", "Toys & Games")]}
    products = {"B": {}, "OLD": {"age_min": 14, "age_max": None}, "OK": {"age_min": 1, "age_max": 3}}
    fake, hits = FakeRanker(), [{"parent_asin": "B", "score": 0.9, "rank": 1}]
    c = GiftConstraints(category_preference="Toys & Games", age_group="toddler").to_handoff()
    out = HybridRanker(fake, config=object())(hits, c, products, top_k=5)
    assert fake.semantic == hits
    assert [(r["parent_asin"], r["rank"]) for r in out] == [("OK", 1)]


def test_semantic_export_format():
    class FakeRetriever:
        def search_batch(self, queries, top_n):
            self.queries, self.top_n = queries, top_n
            return [[{"parent_asin": "A", "score": 0.5, "rank": 1}]] * len(queries)
    r = FakeRetriever()
    out = export([{"qid": "q1", "constraints": {"recipient": "nephew", "recipient_age": 7,
                                                "interests": ["dinosaur"], "budget_max": 30}}], r)
    assert out == {"q1": [["A", 0.5]]}
    assert r.top_n == 200
    assert r.queries == ["dinosaur"]
