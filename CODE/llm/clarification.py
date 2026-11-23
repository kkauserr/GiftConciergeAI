"""Decide whether to ask the user one targeted question before recommending.

Priority:
  1. A contradiction from this turn (from dialogue_state): always asked.
  2. Missing interests: asked once.
  3. Missing budget (and the user hasn't said "no limit"): asked once.

Each missing slot is asked about at most once (tracked in state.asked_slots).
If the user skips the question, we recommend with what we have rather than
nagging. Questions are deterministic templates (no LLM call), so they are
testable and cheap.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClarifyingQuestion:
    slot: str        # "interests", "budget", or "conflict:<kind>"
    question: str


def _recipient_phrase(recipient):
    if not recipient:
        return None
    text = recipient.strip()
    lowered = text.lower()
    for mine, yours in (("my ", "your "), ("our ", "your ")):
        if lowered.startswith(mine):
            return yours + text[len(mine):]
    return text


def interests_question(constraints):
    who = _recipient_phrase(constraints.recipient)
    if who:
        return (f"What is {who} into? A couple of hobbies or interests "
                "will help me find something that fits.")
    return ("Who is the gift for, and what are they into? A couple of hobbies "
            "or interests will help me find something that fits.")


def budget_question(constraints):
    who = _recipient_phrase(constraints.recipient)
    target = f"the gift for {who}" if who else "this gift"
    return (f"What's your budget for {target}? A range like $20-$50 is fine, "
            "or say 'no limit'.")


def missing_slots(constraints):
    missing = []
    if not constraints.interests:
        missing.append("interests")
    if not constraints.has_budget() and not constraints.budget_flexible:
        missing.append("budget")
    return missing


def next_question(state, update_result=None) -> Optional[ClarifyingQuestion]:
    """Return the one question to ask now, or None if we can recommend."""
    if update_result is not None and update_result.conflicts:
        conflict = update_result.conflicts[0]
        return ClarifyingQuestion(f"conflict:{conflict.kind}", conflict.question)

    for slot in missing_slots(state.constraints):
        if slot in state.asked_slots:
            continue
        state.asked_slots.add(slot)
        if slot == "interests":
            return ClarifyingQuestion(slot, interests_question(state.constraints))
        return ClarifyingQuestion(slot, budget_question(state.constraints))
    return None
