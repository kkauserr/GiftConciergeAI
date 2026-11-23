"""Clarifying-question selection (no API calls)."""

from llm.clarification import next_question
from llm.constraints import TurnDelta
from llm.dialogue_state import DialogueState


def step(state, **delta):
    result = state.apply_delta(TurnDelta(**delta), "")
    return next_question(state, result)


def test_asks_interests_first_then_budget_then_recommends():
    state = DialogueState()
    q1 = step(state, recipient="my sister")
    assert q1.slot == "interests" and "your sister" in q1.question
    q2 = step(state, interests_add=["yoga"])
    assert q2.slot == "budget" and "your sister" in q2.question
    assert step(state, budget_max=50) is None


def test_asks_only_one_question_per_turn():
    state = DialogueState()
    q = step(state)
    assert q.slot == "interests"
    assert "Who is the gift for" in q.question


def test_each_slot_asked_at_most_once():
    state = DialogueState()
    assert step(state).slot == "interests"
    assert step(state).slot == "budget"   # user skipped interests; move on
    assert step(state) is None            # both skipped; recommend anyway


def test_no_budget_question_when_flexible_or_known():
    state = DialogueState()
    assert step(state, interests_add=["golf"], no_budget_preference=True) is None
    state = DialogueState()
    assert step(state, interests_add=["golf"], budget_min=30) is None


def test_conflict_question_takes_priority_and_repeats():
    state = DialogueState()
    q = step(state, interests_add=["lego"], budget_min=90, budget_max=20)
    assert q.slot == "conflict:budget_min_gt_max"
    q = step(state, budget_min=80, budget_max=10)
    assert q.slot == "conflict:budget_min_gt_max"  # still asked: never guessed
    assert step(state, budget_max=40) is None
