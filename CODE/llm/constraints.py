"""Constraint schema, LLM extraction, and retrieval-query construction.

Two models:

* TurnDelta: what the LLM extracts from ONE user message (only what that
  message says). It is produced with strict JSON-schema output and then
  validated with pydantic.
* GiftConstraints: the accumulated state after applying deltas with the
  deterministic rules in dialogue_state.py. Its to_handoff() dict is the
  "constraints JSON" passed to ranking (Disha) and explanation.

Keeping merge logic out of the LLM makes the conflict rules testable without
API calls.
"""

import json
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm.exclusions import EXCLUSION_GROUPS, normalize_exclusions
from llm.interest_labels import interest_labels, labels_for, prompt_hints

log = logging.getLogger(__name__)

AgeGroup = Literal["baby", "toddler", "child", "teen", "adult", "senior"]
Category = Literal["Books", "Handmade Products", "Toys & Games"]
BudgetDirection = Literal["cheaper", "more_expensive"]

AGE_GROUPS = list(AgeGroup.__args__)
CATEGORIES = list(Category.__args__)

# Approximate age ranges (years) for filtering against product age_min/age_max.
AGE_GROUP_RANGES = {
    "baby": (0, 1),
    "toddler": (1, 3),
    "child": (4, 12),
    "teen": (13, 17),
    "adult": (18, None),
    "senior": (65, None),
}


def _clean_terms(values):
    """Lowercase, strip, drop empties, dedupe while keeping order."""
    seen, out = set(), []
    for value in values or []:
        term = " ".join(str(value).split()).strip().lower()
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def _clean_text(value):
    if value is None:
        return None
    value = " ".join(str(value).split()).strip()
    return value or None


def short_recipient(recipient):
    """"my 8 year old nephew" -> "nephew" (the shared contract's form)."""
    if not recipient:
        return None
    text = re.sub(r"^(?:my|our|the|an?)\s+", "", recipient.strip(), flags=re.I)
    text = re.sub(r"^\d+(?:\.\d+)?[- ](?:years?|yrs?|months?)[- ]old\s+", "", text, flags=re.I)
    return text or None


class InterestLabel(BaseModel):
    """KG label the LLM assigned to one interest (see interest_labels.py)."""

    model_config = ConfigDict(extra="forbid")

    interest: str
    label: str


class TurnDelta(BaseModel):
    """Constraints stated in a single user message."""

    model_config = ConfigDict(extra="forbid")

    recipient: Optional[str] = None
    occasion: Optional[str] = None
    interests_add: list[str] = Field(default_factory=list)
    interest_labels: list[InterestLabel] = Field(default_factory=list)
    interests_remove: list[str] = Field(default_factory=list)
    replace_interests: bool = False
    exclusions_add: list[str] = Field(default_factory=list)
    budget_min: Optional[float] = Field(default=None, ge=0)
    budget_max: Optional[float] = Field(default=None, ge=0)
    budget_relative: Optional[BudgetDirection] = None
    budget_relative_amount: Optional[float] = Field(default=None, gt=0)
    no_budget_preference: bool = False
    age: Optional[float] = Field(default=None, ge=0, le=120)
    age_group: Optional[AgeGroup] = None
    category_preference: Optional[Category] = None

    @field_validator("interests_add", "interests_remove", "exclusions_add")
    @classmethod
    def _clean_lists(cls, value):
        return _clean_terms(value)

    @field_validator("recipient", "occasion")
    @classmethod
    def _clean_strings(cls, value):
        return _clean_text(value)

    def is_empty(self):
        return self == TurnDelta()


class GiftConstraints(BaseModel):
    """Accumulated constraints for the conversation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    recipient: Optional[str] = None
    occasion: Optional[str] = None
    interests: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    budget_min: Optional[float] = Field(default=None, ge=0)
    budget_max: Optional[float] = Field(default=None, ge=0)
    budget_flexible: bool = False
    age: Optional[float] = Field(default=None, ge=0, le=120)
    age_group: Optional[AgeGroup] = None
    category_preference: Optional[Category] = None
    # {interest: KG label} from extraction; bookkeeping, not a user constraint.
    interest_label_map: dict[str, str] = Field(default_factory=dict, exclude=True)

    def has_budget(self):
        return self.budget_min is not None or self.budget_max is not None

    def target_age_range(self):
        """[min, max] years for age filtering, or None if age is unknown."""
        if self.age is not None:
            return [self.age, self.age]
        if self.age_group:
            return list(AGE_GROUP_RANGES[self.age_group])
        return None

    def to_handoff(self, clarification_question=None):
        """Shared constraints JSON (ranking, KG, frontend, explanation).

        First block: the agreed contract. Second block: our extra keys.
        `exclusions` is normalized for the ranking filter; show
        `exclusions_stated` to users.
        """
        age = self.age
        if age is not None and float(age).is_integer():
            age = int(age)
        return {
            "recipient": short_recipient(self.recipient),
            "recipient_age": age,
            "occasion": self.occasion,
            "interests": list(self.interests),
            "keywords": [],
            "budget_min": self.budget_min,
            "budget_max": self.budget_max,
            "exclusions": normalize_exclusions(self.exclusions),
            "needs_clarification": clarification_question is not None,
            "clarification_question": clarification_question,

            "exclusions_stated": list(self.exclusions),
            "interest_labels": labels_for(self.interests, self.interest_label_map),
            "category_preference": self.category_preference,
            "age_group": self.age_group,
            "target_age_range": self.target_age_range(),
            "budget_flexible": self.budget_flexible,
        }


def _nullable(schema):
    out = dict(schema)
    out["type"] = [schema["type"], "null"]
    if "enum" in out:
        out["enum"] = list(out["enum"]) + [None]
    return out


_STR_LIST = {"type": "array", "items": {"type": "string"}}

# Strict structured-output schema (all keys required, nullable where optional).
# Kept by hand so it stays strict-mode compatible; test_constraints checks it
# matches the TurnDelta fields.
TURN_DELTA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recipient": _nullable({"type": "string"}),
        "occasion": _nullable({"type": "string"}),
        "interests_add": _STR_LIST,
        "interest_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["interest", "label"],
                "properties": {
                    "interest": {"type": "string"},
                    # enum comes from the KG vocabulary when it is available
                    "label": {"type": "string", **({"enum": interest_labels()}
                                                   if interest_labels() else {})},
                },
            },
        },
        "interests_remove": _STR_LIST,
        "replace_interests": {"type": "boolean"},
        "exclusions_add": _STR_LIST,
        "budget_min": _nullable({"type": "number"}),
        "budget_max": _nullable({"type": "number"}),
        "budget_relative": _nullable({"type": "string", "enum": ["cheaper", "more_expensive"]}),
        "budget_relative_amount": _nullable({"type": "number"}),
        "no_budget_preference": {"type": "boolean"},
        "age": _nullable({"type": "number"}),
        "age_group": _nullable({"type": "string", "enum": AGE_GROUPS}),
        "category_preference": _nullable({"type": "string", "enum": CATEGORIES}),
    },
}
TURN_DELTA_SCHEMA["required"] = list(TURN_DELTA_SCHEMA["properties"])

EXTRACTION_SYSTEM_PROMPT = """\
You extract gift-shopping constraints from the user's LATEST message for a gift \
recommender covering three Amazon categories: Books, Handmade Products, Toys & Games.

Return ONLY what the latest message states or changes. Use null / [] / false for \
anything it does not mention. The current state is given so you can resolve \
references ("she", "that", "same budget") and understand short answers to the \
assistant's last question; do not copy unchanged values from it.

Fields:
- recipient: who the gift is for, e.g. "my sister", "7 year old nephew", "coworker".
- occasion: e.g. "birthday", "christmas", "graduation", "housewarming".
- interests_add: hobbies/likes/themes newly mentioned, short lowercase noun phrases \
("dinosaurs", "mystery novels", "knitting").
- interest_labels: for each entry of interests_add that clearly fits one of the labels \
below, {"interest": <exact interests_add string>, "label": <label>}. Keep the specific \
interest in interests_add ("star wars") and give its label here (scifi_fantasy). Leave \
out interests that fit no label; never force the closest one. "Reading" has no label.
- interests_remove: interests from the current state the user takes back or says \
the recipient no longer likes ("she's not into puzzles anymore" -> ["puzzles"]).
- replace_interests: true only if the user replaces the old interests \
("instead", "forget X, she likes Y", "actually only Y").
- exclusions_add: things to avoid, short lowercase noun phrases ("no books" -> "books", \
"no mugs" -> "mugs"). When the user rules out a whole kind of product, use the exact \
group name: <<GROUPS>>. "nothing wearable" / "nothing to wear" -> "wearable"; "no clothes" \
-> "clothing"; "nothing with batteries" -> "electronic"; "hates plush toys" -> "stuffed animals".
- budget_min / budget_max: USD. "under $50" -> max 50. "at least $30" -> min 30. \
"$20-40" -> 20 and 40. "around $50" -> min 40, max 60. A single new amount such as \
"actually $150" or "my budget is $150" -> max 150.
- budget_relative: "cheaper" / "more_expensive" when the user asks to move the price \
relative to before WITHOUT giving a new absolute budget. budget_relative_amount is the \
dollar change if stated ("$20 cheaper" -> 20), else null.
- no_budget_preference: true if the user says budget doesn't matter / no limit.
- age: recipient age in years if stated (months -> fraction of a year).
- age_group: one of baby, toddler, child, teen, adult, senior if stated or clearly implied.
- category_preference: one of the three categories only if the user asks for that kind \
of product ("a book", "a toy or game", "something handmade").

If the message contradicts itself, still extract exactly what it says (e.g. both a \
min above the max); the system will ask the user to clarify.

Interest labels (label: example phrasings):
<<LABELS>>"""
EXTRACTION_SYSTEM_PROMPT = (
    EXTRACTION_SYSTEM_PROMPT
    .replace("<<GROUPS>>", ", ".join(f'"{g}"' for g in EXCLUSION_GROUPS))
    .replace("<<LABELS>>", prompt_hints() or "  (none available: return [])"))


class ExtractionResult(BaseModel):
    delta: TurnDelta
    raw: Optional[dict] = None
    attempts: int = 1
    error: Optional[str] = None


def build_extraction_messages(message, state=None, last_assistant=None):
    context = {
        "current_state": state.model_dump() if state is not None else GiftConstraints().model_dump(),
        "assistant_last_message": last_assistant,
    }
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content":
            "Context:\n" + json.dumps(context, indent=2)
            + "\n\nLatest user message:\n" + message},
    ]


def extract_turn(llm, message, state=None, last_assistant=None, max_attempts=2):
    """Extract a validated TurnDelta from one user message.

    Retries once with the validation error if the output fails validation;
    if it still fails, returns an empty delta with `error` set (the dialogue
    manager will then ask for clarification instead of guessing).
    """
    messages = build_extraction_messages(message, state, last_assistant)
    error = None
    raw = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm.complete_json(messages, TURN_DELTA_SCHEMA, "gift_constraints_delta")
            return ExtractionResult(delta=TurnDelta.model_validate(raw), raw=raw, attempts=attempt)
        except ValidationError as exc:
            error = f"validation failed: {exc.errors(include_url=False)}"
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw)},
                {"role": "user", "content":
                    f"That output was invalid ({error}). Return corrected JSON."},
            ]
        except Exception as exc:  # provider/network/JSON errors
            error = f"{type(exc).__name__}: {exc}"
        log.warning("constraint extraction attempt %d failed: %s", attempt, error)
    return ExtractionResult(delta=TurnDelta(), raw=raw, attempts=max_attempts, error=error)


def build_query_text(constraints):
    """Semantic retrieval query from the constraints.

    Just the interests when there are any: on the 12 dev queries, recipient, age,
    occasion and the word "gift" pulled in party supplies and birthday decorations
    (semantic-only nDCG@10 0.480 -> 0.630 without them). Adding product words
    ("dinosaur toys, dinosaur books") also hurt. Without interests we fall back to
    a sentence. An explicit category is kept in both cases.

    Budget and exclusions are deliberately left out: they are hard filters for
    the ranking stage, and negations ("no books") hurt dense retrieval.
    """
    if constraints.interests:
        query = ", ".join(constraints.interests)
        noun = {"Books": "books", "Toys & Games": "toys and games",
                "Handmade Products": "handmade"}.get(constraints.category_preference)
        return f"{query}, {noun}" if noun else query

    parts = []
    who = constraints.recipient or "someone"
    if constraints.age is not None:
        age = int(constraints.age) if float(constraints.age).is_integer() else constraints.age
        who += f", age {age}"
    elif constraints.age_group:
        who += f" ({constraints.age_group})"
    gift = "gift"
    if constraints.category_preference == "Books":
        gift = "book gift"
    elif constraints.category_preference == "Toys & Games":
        gift = "toy or game gift"
    elif constraints.category_preference == "Handmade Products":
        gift = "handmade gift"
    parts.append(f"{gift} for {who}")
    if constraints.occasion:
        parts.append(f"for {constraints.occasion}")
    return " ".join(parts)
