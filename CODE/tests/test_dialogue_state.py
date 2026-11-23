"""Multi-turn state update and conflict rules (pure Python, no API calls)."""

import json

import pytest

from llm.constraints import TurnDelta
from llm.dialogue_state import DialogueState


def turn(state, utterance="", **delta):
    return state.apply_delta(TurnDelta(**delta), utterance)


def rules(result):
    return [(c.field, c.rule) for c in result.changes]


@pytest.fixture
def state():
    return DialogueState()


# ---------- budget ----------

def test_newer_explicit_budget_overrides(state):
    turn(state, "under $50", budget_max=50)
    r = turn(state, "actually $150", budget_max=150)
    assert state.constraints.budget_max == 150
    assert rules(r) == [("budget_max", "explicit_override")]
    assert r.changes[0].old == 50 and r.changes[0].utterance == "actually $150"


def test_newer_max_clears_inconsistent_older_min(state):
    turn(state, budget_min=80, budget_max=100)
    r = turn(state, "under $50", budget_max=50)
    assert (state.constraints.budget_min, state.constraints.budget_max) == (None, 50)
    assert ("budget_min", "cleared_inconsistent_older_value") in rules(r)


def test_newer_min_clears_inconsistent_older_max(state):
    turn(state, budget_max=50)
    turn(state, "at least $80", budget_min=80)
    assert (state.constraints.budget_min, state.constraints.budget_max) == (80, None)


def test_consistent_older_bound_is_kept(state):
    turn(state, budget_min=20, budget_max=40)
    turn(state, budget_max=60)
    assert (state.constraints.budget_min, state.constraints.budget_max) == (20, 60)


def test_cheaper_scales_existing_budget(state):
    turn(state, budget_min=40, budget_max=100)
    r = turn(state, "something cheaper", budget_relative="cheaper")
    assert (state.constraints.budget_min, state.constraints.budget_max) == (30, 75)
    assert {rule for _, rule in rules(r)} == {"relative_cheaper"}


def test_more_expensive_scales_existing_budget(state):
    turn(state, budget_max=100)
    turn(state, "more expensive", budget_relative="more_expensive")
    assert state.constraints.budget_max == 133


def test_cheaper_by_stated_amount(state):
    turn(state, budget_max=100)
    turn(state, "$20 cheaper", budget_relative="cheaper", budget_relative_amount=20)
    assert state.constraints.budget_max == 80


def test_cheaper_without_prior_budget_asks(state):
    r = turn(state, "cheaper", budget_relative="cheaper")
    assert r.needs_confirmation
    assert r.conflicts[0].kind == "budget_relative_without_reference"
    assert not state.constraints.has_budget()


def test_cheaper_below_zero_asks(state):
    turn(state, budget_max=10)
    r = turn(state, "$20 cheaper", budget_relative="cheaper", budget_relative_amount=20)
    assert r.conflicts[0].kind == "budget_relative_nonpositive"
    assert state.constraints.budget_max == 10


def test_cheaper_with_explicit_lower_value_uses_explicit(state):
    turn(state, budget_max=100)
    turn(state, "cheaper, like $60", budget_relative="cheaper", budget_max=60)
    assert state.constraints.budget_max == 60


def test_cheaper_with_explicit_higher_value_is_contradiction(state):
    turn(state, budget_max=100)
    r = turn(state, "cheaper, $200", budget_relative="cheaper", budget_max=200)
    assert r.conflicts[0].kind == "budget_relative_vs_explicit"
    assert state.constraints.budget_max == 100


def test_same_turn_min_above_max_asks_but_applies_other_fields(state):
    r = turn(state, "for my mom, at least $80 but under $50",
             recipient="my mom", budget_min=80, budget_max=50)
    assert r.conflicts[0].kind == "budget_min_gt_max"
    assert "$80" in r.conflicts[0].question and "$50" in r.conflicts[0].question
    assert state.constraints.recipient == "my mom"
    assert not state.constraints.has_budget()


def test_no_budget_preference_and_later_explicit_budget(state):
    turn(state, budget_max=50)
    turn(state, "budget doesn't matter", no_budget_preference=True)
    assert state.constraints.budget_flexible and not state.constraints.has_budget()
    turn(state, "ok, under $30", budget_max=30)
    assert not state.constraints.budget_flexible and state.constraints.budget_max == 30


def test_no_budget_with_value_same_turn_is_contradiction(state):
    r = turn(state, no_budget_preference=True, budget_max=40)
    assert r.conflicts[0].kind == "budget_no_pref_with_value"
    assert not state.constraints.has_budget() and not state.constraints.budget_flexible


# ---------- interests / exclusions ----------

def test_new_interests_accumulate_and_dedupe(state):
    turn(state, interests_add=["dinosaurs"])
    turn(state, interests_add=["space", "Dinosaur"])
    assert state.constraints.interests == ["dinosaurs", "space"]


def test_instead_replaces_without_excluding(state):
    turn(state, interests_add=["puzzles", "lego"])
    r = turn(state, "actually cooking instead", interests_add=["cooking"], replace_interests=True)
    assert state.constraints.interests == ["cooking"]
    assert state.constraints.exclusions == []
    assert rules(r) == [("interests", "replaced_instead")]


def test_removed_interest_goes_to_exclusions(state):
    turn(state, interests_add=["puzzles", "lego"])
    r = turn(state, "she's not into puzzles anymore", interests_remove=["puzzle"])
    assert state.constraints.interests == ["lego"]
    assert state.constraints.exclusions == ["puzzle"]
    assert rules(r) == [("interests", "interest_removed"),
                        ("exclusions", "removed_interest_excluded")]


def test_readding_excluded_interest_removes_exclusion(state):
    turn(state, exclusions_add=["puzzles"])
    r = turn(state, "actually puzzles are fine, she loves them", interests_add=["puzzles"])
    assert state.constraints.interests == ["puzzles"]
    assert state.constraints.exclusions == []
    assert ("exclusions", "readded_overrides_exclusion") in rules(r)


def test_excluding_current_interest_removes_it(state):
    turn(state, interests_add=["video games", "art"])
    turn(state, "no video games", exclusions_add=["video games"])
    assert state.constraints.interests == ["art"]
    assert state.constraints.exclusions == ["video games"]


def test_interest_added_and_excluded_same_turn_asks(state):
    r = turn(state, "she likes horses, no horses", interests_add=["horses"],
             exclusions_add=["horses"], recipient="niece")
    assert r.conflicts[0].kind == "interest_added_and_removed"
    assert state.constraints.interests == [] and state.constraints.exclusions == []
    assert state.constraints.recipient == "niece"


def test_conflict_on_one_interest_does_not_block_others(state):
    turn(state, interests_add=["horses", "drawing"], exclusions_add=["horses"])
    assert state.constraints.interests == ["drawing"]
    assert state.constraints.exclusions == []


# ---------- category / age / scalars ----------

def test_excluding_preferred_category_clears_it(state):
    turn(state, category_preference="Books")
    turn(state, "no books", exclusions_add=["books"])
    assert state.constraints.category_preference is None
    assert state.constraints.exclusions == ["books"]


def test_preferring_excluded_category_clears_exclusion(state):
    turn(state, exclusions_add=["books"])
    turn(state, "ok, a book then", category_preference="Books")
    assert state.constraints.category_preference == "Books"
    assert state.constraints.exclusions == []


def test_category_and_its_exclusion_same_turn_asks(state):
    r = turn(state, category_preference="Books", exclusions_add=["books"])
    assert r.conflicts[0].kind == "category_excluded_same_turn"
    assert state.constraints.category_preference is None


def test_new_age_clears_inconsistent_age_group(state):
    turn(state, age_group="child")
    r = turn(state, "she's 15", age=15)
    assert state.constraints.age == 15 and state.constraints.age_group is None
    assert ("age_group", "cleared_inconsistent_older_value") in rules(r)


def test_age_and_group_mismatch_same_turn_asks(state):
    r = turn(state, age=30, age_group="toddler")
    assert r.conflicts[0].kind == "age_mismatch"
    assert state.constraints.age is None


def test_recipient_override_and_noop(state):
    turn(state, recipient="my dad")
    r = turn(state, recipient="my uncle")
    assert rules(r) == [("recipient", "explicit_override")]
    assert turn(state, recipient="my uncle").changes == []


# ---------- logging ----------

def test_every_change_is_logged(tmp_path):
    log_file = tmp_path / "logs" / "state.jsonl"
    state = DialogueState(log_path=log_file)
    turn(state, "for my nephew, likes lego, under $40",
         recipient="my nephew", interests_add=["lego"], budget_max=40)
    turn(state, "cheaper", budget_relative="cheaper")
    turn(state, "between $50 and $20", budget_min=50, budget_max=20)

    assert [(c.turn, c.field) for c in state.history] == [
        (1, "recipient"), (1, "interests"), (1, "budget_max"), (2, "budget_max")]
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in records] == ["change"] * 4 + ["conflict"]
    assert records[3] == {**records[3], "old": 40, "new": 30, "rule": "relative_cheaper",
                          "utterance": "cheaper", "turn": 2}
    assert records[4]["kind"] == "budget_min_gt_max" and records[4]["turn"] == 3
    assert state.to_dict()["turn"] == 3
