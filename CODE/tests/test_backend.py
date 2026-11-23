"""Backend layer (CODE/backend): engine wiring, response assembly, sessions.

Runs the real GiftConcierge with the offline LLM on the 900-product sample, so
no credentials, FAISS index or BM25 index are needed.
"""

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CODE.backend.assemble import _highlight, to_ui_response   # noqa: E402
from CODE.backend.config import Settings                        # noqa: E402
from CODE.backend.engine import Engine, EngineUnavailable        # noqa: E402
from CODE.backend.sessions import SessionStore                  # noqa: E402

SAMPLE = REPO_ROOT / "DATA" / "samples" / "sample_corpus_v2.jsonl"
CARD_FIELDS = {"parent_asin", "rank", "title", "price", "rating", "rating_count", "category",
               "image_url", "amazon_url", "why_it_fits", "review_evidence"}


def _sample_asins(n):
    with open(SAMPLE, encoding="utf-8") as infile:
        return [json.loads(line)["parent_asin"] for _, line in zip(range(n), infile)]


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    lookup = tmp_path_factory.mktemp("images") / "lookup.json"
    lookup.write_text(json.dumps({a: f"https://example.org/{a}.jpg" for a in _sample_asins(900)}))
    settings = Settings(llm="offline", corpus=SAMPLE, retriever="tfidf",
                        rank_index=REPO_ROOT / "no-such-index", kg="off", image_lookup=lookup)
    return Engine(settings)


def test_engine_reports_components(engine):
    status = engine.status
    assert status["products"] == 900
    assert status["llm"].startswith("offline")
    assert status["retriever"].startswith("tfidf")
    assert status["ranker"].startswith("stub")
    assert status["kg"] == "off"
    assert status["images"].startswith("900")


def test_conversation_produces_cards(engine):
    concierge = engine.new_conversation()

    first = to_ui_response(concierge.handle("gift for my 8 year old nephew"), engine)
    assert first["type"] == "question" and first["message"]

    concierge.handle("he loves dinosaurs and space")
    results = to_ui_response(concierge.handle("under $30"), engine)
    assert results["type"] == "recommendations"
    assert results["constraints"]["budget_max"] == 30
    assert 0 < len(results["recommendations"]) <= 5

    for card in results["recommendations"]:
        assert set(card) == CARD_FIELDS
        assert card["price"] is None or card["price"] <= 30
        assert card["amazon_url"] == f"https://www.amazon.com/dp/{card['parent_asin']}"
        assert card["image_url"] == f"https://example.org/{card['parent_asin']}.jpg"
        assert card["title"] and card["why_it_fits"]


def test_refinement_updates_state_and_message(engine):
    concierge = engine.new_conversation()
    concierge.handle("birthday gift for my nephew who loves dinosaurs, under $40")
    cheaper = to_ui_response(concierge.handle("cheaper"), engine, previous_results=True)
    assert cheaper["constraints"]["budget_max"] == 30
    assert cheaper["message"].startswith("Here's an updated set")


def test_internal_fields_only_in_debug(engine):
    concierge = engine.new_conversation()
    raw = concierge.handle("graduation gift for my sister who loves coffee and reading, under $80")
    assert raw["type"] == "recommendations"

    public = json.dumps(to_ui_response(raw, engine))
    for internal in ("evidence_ids", "used_evidence", "invalid_citations", "query_text", ".title", "score"):
        assert internal not in public

    debug = to_ui_response(raw, engine, debug=True)["debug"]
    assert debug["query_text"] and debug["explanations"]


def test_highlight_prefers_reviews_and_skips_cautions():
    review = {"field": "review_samples", "text": "5/5 review: Great gift - She uses it daily"}
    kg_claim = {"field": "kg", "text": "Reviewers often mention it is durable."}
    caution = {"field": "kg", "text": "Caution: some reviewers mention small parts."}

    assert _highlight({"used_evidence": [kg_claim, review]}) == {"text": "Great gift - She uses it daily",
                                                                 "source": "review"}
    assert _highlight({"used_evidence": [caution, kg_claim]})["source"] == "insight"
    assert _highlight({"used_evidence": [caution]}) is None
    assert _highlight({}) is None


def test_image_lookup_value_formats(engine):
    saved = engine.images
    engine.images = {"A": "https://x/a.jpg", "B": {"large": "https://x/b.jpg"},
                     "C": [{"hi_res": None, "large": "https://x/c.jpg"}], "D": "", "E": None}
    try:
        assert [engine.image_for(k) for k in "ABCDEF"] == [
            "https://x/a.jpg", "https://x/b.jpg", "https://x/c.jpg", None, None, None]
    finally:
        engine.images = saved


def _no_credentials(monkeypatch):
    """Simulate a machine without FOUNDRY_* credentials, even if a developer has a .env."""
    from llm import llm_client

    def from_env(*args, **kwargs):
        raise llm_client.LLMConfigError("Missing FOUNDRY_* (simulated)")

    monkeypatch.setattr(llm_client.LLMClient, "from_env", from_env)


def test_auto_llm_falls_back_to_offline_without_credentials(monkeypatch):
    _no_credentials(monkeypatch)
    settings = Settings(llm="auto", corpus=SAMPLE, retriever="tfidf",
                        rank_index=REPO_ROOT / "no-such-index", kg="off")
    engine = Engine(settings)
    assert engine.status["llm"].startswith("offline (auto")
    reply = engine.new_conversation().handle("gift for my nephew who loves dinosaurs, under $30")
    assert reply["type"] == "recommendations"


def test_foundry_mode_requires_credentials(monkeypatch):
    _no_credentials(monkeypatch)
    with pytest.raises(EngineUnavailable, match="FOUNDRY"):
        Engine(Settings(llm="foundry", corpus=SAMPLE, retriever="tfidf", kg="off"))


def test_missing_corpus_is_reported(tmp_path):
    with pytest.raises(EngineUnavailable, match="corpus not found"):
        Engine(Settings(llm="offline", corpus=tmp_path / "missing.jsonl", kg="off"))


def test_session_store_reuses_expires_and_caps():
    made = []
    store = SessionStore(lambda: made.append(1) or object(), ttl_seconds=0.05, max_sessions=2)

    first = store.get("a")
    assert store.get("a") is first and len(made) == 1

    store.get("b")
    store.get("c")                      # over the cap: "a" (least recent) is evicted
    assert len(store) == 2 and store.get("a") is not first

    time.sleep(0.08)
    store.get("z")                      # everything else has expired
    assert len(store) == 1

    store.drop("z")
    assert len(store) == 0
