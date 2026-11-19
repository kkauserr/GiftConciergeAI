"""Constraint schema, validation, extraction retries and query building (no API calls)."""

import pytest
from pydantic import ValidationError

from llm.constraints import (
    TURN_DELTA_SCHEMA,
    GiftConstraints,
    TurnDelta,
    build_query_text,
    extract_turn,
)
from llm.llm_client import FakeLLMClient


def full_delta(**overrides):
    """A complete strict-schema response with every key present."""
    base = {name: None for name in TURN_DELTA_SCHEMA["properties"]}
    base.update(interests_add=[], interests_remove=[], exclusions_add=[],
                replace_interests=False, no_budget_preference=False)
    base.update(overrides)
    return base


# ---------- schema ----------

def _assert_strict(schema, path="root"):
    if schema.get("type") == "object" or "properties" in schema:
        assert schema.get("additionalProperties") is False, path
        assert set(schema["required"]) == set(schema["properties"]), path
        for key, sub in schema["properties"].items():
            _assert_strict(sub, f"{path}.{key}")
    if "items" in schema:
        _assert_strict(schema["items"], path + "[]")


def test_schema_is_strict_mode_compatible():
    _assert_strict(TURN_DELTA_SCHEMA)


def test_schema_matches_pydantic_model():
    assert set(TURN_DELTA_SCHEMA["properties"]) == set(TurnDelta.model_fields)


def test_schema_enums_match_model():
    assert TURN_DELTA_SCHEMA["properties"]["age_group"]["enum"][:-1] == \
        ["baby", "toddler", "child", "teen", "adult", "senior"]
    assert TURN_DELTA_SCHEMA["properties"]["category_preference"]["enum"][:-1] == \
        ["Books", "Handmade Products", "Toys & Games"]


# ---------- validation ----------

def test_valid_delta_is_normalized():
    delta = TurnDelta.model_validate(full_delta(
        recipient="  my   nephew ", interests_add=["Dinosaurs", "dinosaurs ", "", "Space"],
        budget_max=50, age=7, age_group="child", category_preference="Toys & Games"))
    assert delta.recipient == "my nephew"
    assert delta.interests_add == ["dinosaurs", "space"]
    assert delta.budget_max == 50


@pytest.mark.parametrize("bad", [
    {"budget_max": -5},
    {"age": 300},
    {"age_group": "toddlerish"},
    {"category_preference": "Electronics"},
    {"budget_relative": "cheaperish"},
    {"budget_relative_amount": 0},
    {"surprise": "extra key"},
    {"interests_add": "dinosaurs"},
])
def test_invalid_delta_rejected(bad):
    with pytest.raises(ValidationError):
        TurnDelta.model_validate(full_delta(**bad))


def test_empty_delta():
    assert TurnDelta.model_validate(full_delta()).is_empty()


def test_constraints_handoff_shape():
    c = GiftConstraints(recipient="niece", interests=["art"], budget_max=30, age_group="child")
    out = c.to_handoff()
    assert set(out) == {"recipient", "occasion", "interests", "exclusions", "budget_min",
                        "budget_max", "budget_flexible", "age", "age_group",
                        "category_preference", "target_age_range"}
    assert out["target_age_range"] == [4, 12]
    assert GiftConstraints(age=7, age_group="child").target_age_range() == [7, 7]


# ---------- extraction ----------

def test_extract_turn_valid_first_try():
    llm = FakeLLMClient([full_delta(interests_add=["Chess"], budget_max=40)])
    state = GiftConstraints(recipient="dad")
    result = extract_turn(llm, "he likes chess, under $40", state=state,
                          last_assistant="What does he like?")
    assert result.attempts == 1 and result.error is None
    assert result.delta.interests_add == ["chess"]
    call = llm.calls[0]
    assert call["schema_name"] == "gift_constraints_delta"
    assert '"recipient": "dad"' in call["messages"][1]["content"]
    assert "What does he like?" in call["messages"][1]["content"]


def test_extract_turn_retries_on_invalid_output():
    llm = FakeLLMClient([full_delta(age_group="kiddo"), full_delta(age_group="child")])
    result = extract_turn(llm, "for a kid")
    assert result.attempts == 2 and result.delta.age_group == "child"
    retry_messages = llm.calls[1]["messages"]
    assert "invalid" in retry_messages[-1]["content"]


def test_extract_turn_gives_empty_delta_after_repeated_failure():
    llm = FakeLLMClient([full_delta(budget_max=-1), full_delta(budget_max=-2)])
    result = extract_turn(llm, "???")
    assert result.delta.is_empty() and result.error


def test_extract_turn_handles_provider_error():
    def boom(kind, messages, name):
        raise RuntimeError("network down")
    result = extract_turn(FakeLLMClient(boom), "hi")
    assert result.delta.is_empty() and "network down" in result.error


# ---------- query text ----------

def test_query_text_uses_preferences_not_filters():
    c = GiftConstraints(recipient="my nephew", age=7, interests=["dinosaurs", "space"],
                        occasion="birthday", budget_max=40, exclusions=["books"],
                        category_preference="Toys & Games")
    q = build_query_text(c)
    assert q == "toy or game gift for my nephew, age 7 who likes dinosaurs, space for birthday"
    assert "40" not in q and "books" not in q


def test_query_text_minimal():
    assert build_query_text(GiftConstraints()) == "gift for someone"
    assert build_query_text(GiftConstraints(age_group="teen", interests=["anime"])) == \
        "gift for someone (teen) who likes anime"
