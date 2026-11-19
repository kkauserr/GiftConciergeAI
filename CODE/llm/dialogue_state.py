"""Multi-turn dialogue state with explicit, deterministic conflict rules.

The LLM only extracts what each message says (a TurnDelta); this module decides
how that changes the accumulated GiftConstraints. Every change is recorded in
`state.history` (and optionally appended to a JSONL log) with the rule that
caused it.

Rules (rule names appear in the change log):

Budget
  explicit_override         A newer explicit budget value replaces the older one
                            ("actually $150" -> budget_max = 150).
  cleared_inconsistent_older_value
                            If a newer explicit bound makes an older bound
                            impossible (new max below old min, or new min above
                            old max), the older bound is cleared: newer wins.
  relative_cheaper / relative_more_expensive
                            "cheaper" / "more expensive" adjust the existing
                            bounds: by the stated amount ("$20 cheaper"), else
                            by CHEAPER_FACTOR / MORE_EXPENSIVE_FACTOR.
  no_budget_preference      "budget doesn't matter" clears both bounds and sets
                            budget_flexible; a later explicit budget unsets it.
Interests / exclusions
  interest_added            New interests are added to the list.
  replaced_instead          "instead" (replace_interests) replaces the list; the
                            dropped interests are NOT excluded.
  interest_removed + removed_interest_excluded
                            A removed interest leaves the list and goes into
                            exclusions.
  readded_overrides_exclusion
                            Re-adding something previously excluded removes the
                            exclusion (newer wins).
  excluded_removes_interest Excluding a current interest removes it.
  exclusion_clears_category Excluding the preferred category clears it.
  category_clears_exclusion Preferring a category removes a matching exclusion.
Other fields
  set / explicit_override   recipient, occasion, age, age_group, category.
  cleared_inconsistent_older_value
                            A new age outside the old age_group's range (or vice
                            versa) clears the older one.

Contradictions WITHIN one message are never guessed. The affected fields are
left unchanged, the rest of the message is applied, and a Conflict with a
question for the user is returned:
  budget_min_gt_max, budget_no_pref_with_value, budget_relative_vs_explicit,
  budget_relative_without_reference, budget_relative_nonpositive,
  interest_added_and_removed, age_mismatch, category_excluded_same_turn.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm.constraints import AGE_GROUP_RANGES, GiftConstraints, TurnDelta

log = logging.getLogger(__name__)

CHEAPER_FACTOR = 0.75
MORE_EXPENSIVE_FACTOR = 1.33

CATEGORY_TERMS = {
    "Books": {"book", "books"},
    "Toys & Games": {"toy", "toys", "game", "games", "toys & games", "toys and games"},
    "Handmade Products": {"handmade", "handmade product", "handmade products", "handmade item"},
}


def term_key(term):
    """Loose match key so 'puzzle' == 'Puzzles' and 'story' == 'stories'."""
    t = " ".join(term.lower().split())
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _contains(terms, term):
    key = term_key(term)
    return any(term_key(t) == key for t in terms)


def _without(terms, term):
    key = term_key(term)
    return [t for t in terms if term_key(t) != key]


def _category_of_term(term):
    for category, words in CATEGORY_TERMS.items():
        if term.lower().strip() in words:
            return category
    return None


def _age_in_group(age, group):
    lo, hi = AGE_GROUP_RANGES[group]
    return age >= lo and (hi is None or age <= hi)


def _money(value):
    return f"${value:,.0f}" if float(value).is_integer() else f"${value:,.2f}"


@dataclass
class StateChange:
    turn: int
    field: str
    old: Any
    new: Any
    rule: str
    utterance: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conflict:
    kind: str
    fields: list
    question: str
    details: dict = field(default_factory=dict)


@dataclass
class UpdateResult:
    turn: int
    changes: list
    conflicts: list

    @property
    def needs_confirmation(self):
        return bool(self.conflicts)


class DialogueState:
    def __init__(self, log_path=None):
        self.constraints = GiftConstraints()
        self.turn = 0
        self.history = []          # list[StateChange]
        self.conflict_log = []     # list[(turn, Conflict)]
        self.asked_slots = set()   # clarification bookkeeping (see clarification.py)
        self.log_path = Path(log_path) if log_path else None

    # ---------- public API ----------

    def apply_delta(self, delta: TurnDelta, utterance: str = "") -> UpdateResult:
        self.turn += 1
        self._utterance = utterance
        self._changes = []
        conflicts = self._find_conflicts(delta)
        blocked = {f for c in conflicts for f in c.fields}

        self._apply_scalars(delta, blocked)
        self._apply_category(delta, blocked)
        self._apply_interests(delta, blocked)
        self._apply_exclusions(delta, blocked)
        conflicts += self._apply_budget(delta, blocked)

        for conflict in conflicts:
            self.conflict_log.append((self.turn, conflict))
            self._write_log({"event": "conflict", "turn": self.turn,
                             "utterance": utterance, **asdict(conflict)})
            log.info("turn %d conflict %s: %s", self.turn, conflict.kind, conflict.question)
        return UpdateResult(turn=self.turn, changes=self._changes, conflicts=conflicts)

    def to_dict(self):
        return {
            "turn": self.turn,
            "constraints": self.constraints.model_dump(),
            "asked_slots": sorted(self.asked_slots),
            "history": [asdict(c) for c in self.history],
        }

    # ---------- change recording ----------

    def _set(self, field_name, new, rule):
        old = getattr(self.constraints, field_name)
        if old == new:
            return
        setattr(self.constraints, field_name, new)
        change = StateChange(self.turn, field_name, old, new, rule, self._utterance)
        self.history.append(change)
        self._changes.append(change)
        self._write_log({"event": "change", **asdict(change)})
        log.info("turn %d %s: %r -> %r (%s)", self.turn, field_name, old, new, rule)

    def _write_log(self, record):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, default=str) + "\n")

    # ---------- same-turn contradictions ----------

    def _find_conflicts(self, d: TurnDelta):
        conflicts = []
        if d.budget_min is not None and d.budget_max is not None and d.budget_min > d.budget_max:
            conflicts.append(Conflict(
                "budget_min_gt_max", ["budget"],
                f"You mentioned at least {_money(d.budget_min)} but no more than "
                f"{_money(d.budget_max)}. What budget should I use?",
                {"budget_min": d.budget_min, "budget_max": d.budget_max}))
        has_explicit = d.budget_min is not None or d.budget_max is not None
        if d.no_budget_preference and (has_explicit or d.budget_relative):
            conflicts.append(Conflict(
                "budget_no_pref_with_value", ["budget"],
                "You said budget doesn't matter but also gave a price. "
                "Should I use a budget, and if so, what is it?"))
        for term in d.interests_add:
            if _contains(d.interests_remove, term) or _contains(d.exclusions_add, term):
                conflicts.append(Conflict(
                    "interest_added_and_removed", [f"interest:{term_key(term)}"],
                    f"Should I look for things related to {term}, or avoid them?",
                    {"term": term}))
        if d.age is not None and d.age_group and not _age_in_group(d.age, d.age_group):
            conflicts.append(Conflict(
                "age_mismatch", ["age"],
                f"You mentioned age {d.age:g} and '{d.age_group}'. How old is the recipient?",
                {"age": d.age, "age_group": d.age_group}))
        if d.category_preference and any(
                _category_of_term(t) == d.category_preference for t in d.exclusions_add):
            conflicts.append(Conflict(
                "category_excluded_same_turn", ["category"],
                f"Do you want {d.category_preference.lower()}, or should I avoid them?",
                {"category": d.category_preference}))
        return conflicts

    # ---------- field updates ----------

    def _override_rule(self, field_name):
        return "set" if getattr(self.constraints, field_name) is None else "explicit_override"

    def _apply_scalars(self, d, blocked):
        for name in ("recipient", "occasion"):
            value = getattr(d, name)
            if value is not None:
                self._set(name, value, self._override_rule(name))
        if "age" in blocked:
            return
        if d.age is not None:
            self._set("age", d.age, self._override_rule("age"))
            group = self.constraints.age_group
            if d.age_group is None and group and not _age_in_group(d.age, group):
                self._set("age_group", None, "cleared_inconsistent_older_value")
        if d.age_group is not None:
            self._set("age_group", d.age_group, self._override_rule("age_group"))
            age = self.constraints.age
            if d.age is None and age is not None and not _age_in_group(age, d.age_group):
                self._set("age", None, "cleared_inconsistent_older_value")

    def _apply_category(self, d, blocked):
        if d.category_preference is None or "category" in blocked:
            return
        self._set("category_preference", d.category_preference,
                  self._override_rule("category_preference"))
        kept = [t for t in self.constraints.exclusions
                if _category_of_term(t) != d.category_preference]
        if kept != self.constraints.exclusions:
            self._set("exclusions", kept, "category_clears_exclusion")

    def _apply_interests(self, d, blocked):
        def ok(term):
            return f"interest:{term_key(term)}" not in blocked

        adds = [t for t in d.interests_add if ok(t)]
        removes = [t for t in d.interests_remove if ok(t)]

        if d.replace_interests and adds:
            self._set("interests", list(adds), "replaced_instead")
        else:
            for term in adds:
                if _contains(self.constraints.interests, term):
                    continue
                self._set("interests", self.constraints.interests + [term], "interest_added")
        for term in adds:
            if _contains(self.constraints.exclusions, term):
                self._set("exclusions", _without(self.constraints.exclusions, term),
                          "readded_overrides_exclusion")

        for term in removes:
            if _contains(self.constraints.interests, term):
                self._set("interests", _without(self.constraints.interests, term),
                          "interest_removed")
            if not _contains(self.constraints.exclusions, term):
                self._set("exclusions", self.constraints.exclusions + [term],
                          "removed_interest_excluded")

    def _apply_exclusions(self, d, blocked):
        for term in d.exclusions_add:
            if f"interest:{term_key(term)}" in blocked:
                continue
            category = _category_of_term(term)
            if category and "category" in blocked:
                continue
            if not _contains(self.constraints.exclusions, term):
                self._set("exclusions", self.constraints.exclusions + [term], "exclusion_added")
            if _contains(self.constraints.interests, term):
                self._set("interests", _without(self.constraints.interests, term),
                          "excluded_removes_interest")
            if category and self.constraints.category_preference == category:
                self._set("category_preference", None, "exclusion_clears_category")

    def _apply_budget(self, d, blocked):
        """Returns any conflicts that depend on the existing budget."""
        if "budget" in blocked:
            return []
        c = self.constraints

        if d.no_budget_preference:
            self._set("budget_min", None, "no_budget_preference")
            self._set("budget_max", None, "no_budget_preference")
            self._set("budget_flexible", True, "no_budget_preference")
            return []

        explicit = d.budget_min is not None or d.budget_max is not None
        if explicit:
            # "cheaper, $200" when the old max was $100 contradicts itself.
            if d.budget_relative and c.budget_max is not None and d.budget_max is not None:
                wrong_way = (d.budget_relative == "cheaper" and d.budget_max > c.budget_max) or \
                            (d.budget_relative == "more_expensive" and d.budget_max < c.budget_max)
                if wrong_way:
                    word = "cheaper" if d.budget_relative == "cheaper" else "more expensive"
                    return [Conflict(
                        "budget_relative_vs_explicit", ["budget"],
                        f"You asked for something {word} but gave {_money(d.budget_max)}, "
                        f"and the current limit is {_money(c.budget_max)}. "
                        "What should the maximum be?",
                        {"old_max": c.budget_max, "new_max": d.budget_max})]
            if d.budget_max is not None:
                self._set("budget_max", d.budget_max, self._override_rule("budget_max"))
                if d.budget_min is None and c.budget_min is not None and c.budget_min > d.budget_max:
                    self._set("budget_min", None, "cleared_inconsistent_older_value")
            if d.budget_min is not None:
                self._set("budget_min", d.budget_min, self._override_rule("budget_min"))
                if d.budget_max is None and c.budget_max is not None and c.budget_max < d.budget_min:
                    self._set("budget_max", None, "cleared_inconsistent_older_value")
            self._set("budget_flexible", False, "explicit_override")
            return []

        if d.budget_relative:
            return self._apply_relative_budget(d.budget_relative, d.budget_relative_amount)
        return []

    def _apply_relative_budget(self, direction, amount):
        c = self.constraints
        word = "cheaper" if direction == "cheaper" else "more expensive"
        if c.budget_min is None and c.budget_max is None:
            return [Conflict(
                "budget_relative_without_reference", ["budget"],
                f"Sure, {word}. What price range should I aim for?",
                {"direction": direction})]

        def adjust(value):
            if value is None:
                return None
            if amount is not None:
                new = value - amount if direction == "cheaper" else value + amount
            else:
                new = value * (CHEAPER_FACTOR if direction == "cheaper" else MORE_EXPENSIVE_FACTOR)
            return round(new, 2)

        new_max = adjust(c.budget_max)
        new_min = adjust(c.budget_min)
        if new_max is not None and new_max <= 0:
            return [Conflict(
                "budget_relative_nonpositive", ["budget"],
                f"Taking {_money(amount)} off a {_money(c.budget_max)} limit leaves nothing. "
                "What is the most you want to spend?",
                {"old_max": c.budget_max, "amount": amount})]
        if new_min is not None:
            new_min = max(new_min, 0.0)
        rule = "relative_cheaper" if direction == "cheaper" else "relative_more_expensive"
        if new_max is not None:
            self._set("budget_max", new_max, rule)
        if new_min is not None:
            self._set("budget_min", new_min, rule)
        return []
