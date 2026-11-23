"""Evidence collection, prompt grounding and citation validation (no API calls)."""

from llm.constraints import GiftConstraints
from llm.explanation import collect_evidence, explain_recommendations
from llm.llm_client import FakeLLMClient
from retrieval.config import SAMPLE_CORPUS
from retrieval.corpus import load_products

PRODUCT_A = {
    "parent_asin": "A1",
    "title": "Dino Dig Excavation Kit",
    "price_num": 24.99,
    "average_rating": 4.6,
    "rating_number": 1532,
    "specific_category": "Science Kits",
    "age_min": 6, "age_max": 12,
    "brand_or_author": "DigCo",
    "features": ["Includes 12 dinosaur fossils", "Chisel and brush included", "", "STEM learning"],
    "review_count": 210, "average_review_rating": 4.5, "verified_review_count": 190,
    "review_samples": [
        {"title": "Fun", "text": "My son loved digging", "rating": 5, "helpful_vote": 1},
        {"title": "Messy", "text": "Lots of dust", "rating": 3, "helpful_vote": 9},
    ],
}
PRODUCT_B = {
    "parent_asin": "B2", "title": "Space Atlas for Kids", "price_num": 15.0,
    "average_rating": 4.8, "rating_number": 40, "features": [],
    "description": ["A tour of the solar system with photos."],
}
PRODUCTS = {"A1": PRODUCT_A, "B2": PRODUCT_B}
RANKED = [{"parent_asin": "B2", "score": 0.5, "rank": 2},
          {"parent_asin": "A1", "score": 0.9, "rank": 1}]
CONSTRAINTS = GiftConstraints(recipient="nephew", age=7, interests=["dinosaurs"], budget_max=30)
KG = {"A1": [{"subject": "Dino Dig Excavation Kit", "relation": "related_to", "object": "paleontology"},
             "Popular gift for ages 6-10"]}


def test_collect_evidence_ids_and_fields():
    ev = collect_evidence(PRODUCT_A, "P1", KG)
    ids = [e.id for e in ev]
    assert ids == ["P1.title", "P1.price", "P1.rating", "P1.category", "P1.age", "P1.brand",
                   "P1.feature[0]", "P1.feature[1]", "P1.feature[2]", "P1.review_stats",
                   "P1.review[0]", "P1.review[1]", "P1.kg[0]", "P1.kg[1]"]
    by_id = {e.id: e for e in ev}
    assert by_id["P1.price"].text == "Price: $24.99"
    assert by_id["P1.rating"].text == "Average rating 4.6/5 from 1,532 ratings"
    assert by_id["P1.age"].text == "Recommended age: 6-12 years"
    assert by_id["P1.review[0]"].text.startswith("3/5 review: Messy")  # most helpful first
    assert by_id["P1.kg[0]"].text == "Dino Dig Excavation Kit related_to paleontology"
    assert by_id["P1.kg[0]"].field == "kg"


def test_description_used_only_without_features():
    ids = [e.id for e in collect_evidence(PRODUCT_B, "P2")]
    assert "P2.description[0]" in ids and not any("feature" in i for i in ids)
    assert not any("description" in e.id for e in collect_evidence(PRODUCT_A, "P1"))


def test_kg_provider_object():
    class Provider:
        def get_facts(self, asin):
            return [{"text": f"fact about {asin}"}]
    ev = collect_evidence(PRODUCT_B, "P1", Provider())
    assert ev[-1].id == "P1.kg[0]" and ev[-1].text == "fact about B2"


def test_explanations_track_used_evidence_and_invalid_citations():
    response = {"explanations": [
        {"product_id": "P1", "explanation": "A dinosaur dig kit for ages 6-12.",
         "evidence_ids": ["P1.title", "P1.age", "P1.feature[0]", "P1.made_up", "P2.title"]},
        {"product_id": "P2", "explanation": "A highly rated space atlas.",
         "evidence_ids": ["P2.title", "P2.rating"]},
    ]}
    llm = FakeLLMClient([response])
    out = explain_recommendations(llm, RANKED, PRODUCTS, CONSTRAINTS, kg_facts=KG)
    first, second = out["explanations"]
    assert (first["parent_asin"], first["rank"]) == ("A1", 1)   # ordered by rank
    assert first["evidence_ids"] == ["P1.title", "P1.age", "P1.feature[0]"]
    assert first["invalid_citations"] == ["P1.made_up", "P2.title"]  # cross-product dropped
    assert first["used_evidence"][2] == {"id": "P1.feature[0]", "parent_asin": "A1",
                                         "field": "features", "text": "Includes 12 dinosaur fossils",
                                         "path_id": None}
    assert second["fallback"] is False and second["invalid_citations"] == []
    assert set(out["evidence"]) == {"P1", "P2"}


def test_prompt_contains_only_given_evidence():
    llm = FakeLLMClient([{"explanations": []}])
    explain_recommendations(llm, RANKED, PRODUCTS, CONSTRAINTS, kg_facts=KG)
    call = llm.calls[0]
    assert call["schema_name"] == "gift_explanations"
    prompt = call["messages"][1]["content"]
    assert "[P1.kg[0]] Dino Dig Excavation Kit related_to paleontology" in prompt
    assert "[P2.description[0]] A tour of the solar system" in prompt
    assert '"interests": [\n    "dinosaurs"\n  ]' in prompt
    assert "Only" in call["messages"][0]["content"] or "ONLY" in call["messages"][0]["content"]


def test_missing_or_failed_explanations_fall_back_to_metadata():
    llm = FakeLLMClient([{"explanations": [
        {"product_id": "P1", "explanation": "Great kit.", "evidence_ids": ["P1.title"]}]}])
    out = explain_recommendations(llm, RANKED, PRODUCTS, CONSTRAINTS)
    fb = out["explanations"][1]
    assert fb["fallback"] is True
    assert fb["explanation"] == "Space Atlas for Kids ($15.00), average rating 4.8/5 from 40 ratings."
    assert fb["evidence_ids"] == ["P2.title", "P2.price", "P2.rating"]

    def boom(kind, messages, name):
        raise RuntimeError("timeout")
    out = explain_recommendations(FakeLLMClient(boom), RANKED, PRODUCTS, CONSTRAINTS)
    assert all(x["fallback"] for x in out["explanations"])


def test_top_k_and_sample_review_evidence():
    sample = load_products(SAMPLE_CORPUS)
    asins = list(sample)[:8]
    ranked = [{"parent_asin": a, "score": 1.0, "rank": i + 1} for i, a in enumerate(asins)]
    out = explain_recommendations(FakeLLMClient([{"explanations": []}]), ranked, sample,
                                  GiftConstraints(), top_k=3)
    assert len(out["explanations"]) == 3
    # works with the sample before and after the review join
    for evs in out["evidence"].values():
        product = sample[evs[0]["parent_asin"]]
        has_reviews = bool(product.get("review_count") or product.get("review_samples"))
        assert any("review" in e["id"] for e in evs) == has_reviews
