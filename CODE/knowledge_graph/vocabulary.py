"""Load and validate the versioned Interest and Aspect vocabularies.

The vocabulary JSON files (vocab/interests.json, vocab/aspects.json) are
the single source of truth for labels and matching rules. This module
compiles them into Concept objects whose Rules the inference extractors
evaluate. Rule semantics are documented in the JSON files' "notes".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import normalize as nz
from .ontology import (
    ASPECT_ATTRIBUTE, ASPECT_QUALITY, EVIDENCE_LISTING_STATED, EVIDENCE_STRUCTURED,
    SOURCE_CATEGORY_MAP, SOURCE_DETAILS, SOURCE_DOMAIN, SOURCE_FEATURE_LEXICON,
    SOURCE_TITLE_LEXICON, NodeType, aspect_id, interest_id,
)

VOCAB_DIR = Path(__file__).resolve().parent / "vocab"
INTERESTS_PATH = VOCAB_DIR / "interests.json"
ASPECTS_PATH = VOCAB_DIR / "aspects.json"

RULE_TYPES = {"category", "details_match", "details_flag", "details_min_number",
              "domain", "title", "features"}
_TEXT_RULES = {"title", "features"}
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class VocabularyError(ValueError):
    pass


# --------------------------------------------------------------------------
# Phrase matching
# --------------------------------------------------------------------------

def _match_text(text: str) -> str:
    """Same-length normalisation for matching: curly apostrophes become
    straight ones, so spans still index the original text."""
    return text.replace("’", "'").replace("‘", "'")


def _compile(phrases: Iterable[str]) -> Optional[re.Pattern]:
    bodies = []
    for p in phrases:
        tokens = [t for t in re.split(r"[\s\-]+", p.strip().lower()) if t]
        if tokens:
            bodies.append(r"[\s\-]+".join(re.escape(t) for t in tokens))
    if not bodies:
        return None
    bodies.sort(key=len, reverse=True)  # longest alternative first
    return re.compile(r"(?<!\w)(?:" + "|".join(bodies) + r")(?!\w)", re.IGNORECASE)


@dataclass
class TextMatch:
    start: int
    end: int
    text: str
    context: str


NEGATORS = {"not", "no", "never", "without", "minus", "instead", "alternative",
            "nor", "isn't", "aren't", "wasn't", "doesn't", "don't"}
_CLAUSE_BREAK = re.compile(r"[.;:!?,|()\[\]【】]")
NEGATION_WINDOW_TOKENS = 4


def negated(text: str, start: int) -> bool:
    """True if one of the 4 tokens before `start`, within the same clause,
    is a negator. Example: "It's not printed on the wood", "alternative to
    plastic". Clause punctuation stops the window, so "No assembly. Wooden
    blocks" is not negated."""
    before = text[max(0, start - 60):start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]
    tokens = before.lower().split()[-NEGATION_WINDOW_TOKENS:]
    return any(t in NEGATORS for t in tokens)


# Component guard (material aspects): wording after the match, in the same
# clause, that says the material is absent, optional or being replaced.
COMPONENT_AFTER = ("not included", "is not included", "are not included",
                   "sold separately", "not include", "replaced", "replacement",
                   "is replaced", "are replaced")
COMPONENT_BEFORE = {"replace", "replaces", "replacing", "replacement", "spare"}
_COMPONENT_AFTER = re.compile(
    r"^\W{0,3}(?:\w+\W+){0,3}?(?:" + "|".join(re.escape(p) for p in COMPONENT_AFTER) + r")(?!\w)",
    re.IGNORECASE,
)


def component_excluded(text: str, start: int, end: int) -> bool:
    """True when the clause around a material match says it is a spare,
    replaced, optional or missing component, e.g. "Cotton is not included",
    "Directly replace Original plastic parts", "a spare set of rubber backs"."""
    before = text[max(0, start - 60):start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]
    if any(t.strip(".,") in COMPONENT_BEFORE for t in before.lower().split()[-NEGATION_WINDOW_TOKENS:]):
        return True
    after = text[end:end + 60]
    brk = _CLAUSE_BREAK.search(after)
    if brk:
        after = after[:brk.start()]
    return bool(_COMPONENT_AFTER.search(after))


_USE_PREFIX = re.compile(r"(?<!\w)(?:for|as)\s+(?:\S+\s+){0,2}$", re.IGNORECASE)


def _use_suggestion_spans(text: str, matcher: "PhraseMatcher") -> List[Tuple[int, int]]:
    """Spans of matcher hits preceded by "for"/"as" (+ up to two words)."""
    t = _match_text(text)
    return [m.span() for m in (matcher._inc.finditer(t) if matcher._inc else [])
            if _USE_PREFIX.search(t[max(0, m.start() - 40):m.start()])]


class PhraseMatcher:
    """Whole-word, case-insensitive phrase matcher with overlap exclusions.

    A candidate match is discarded when:
      - it overlaps a match of any `exclude` phrase ("dog" inside "hot dog")
      - it is negated within its clause ("not printed on wood")
      - the keyword-list guard is on and it falls inside a short item of an
        SEO keyword list, a long list of compatible uses, or a compatibility
        clause ("Fit for TCG Yugioh card")
      - it overlaps a caller-supplied `blocked` span (cross-sell sentences,
        kit contents lists)
      - the component guard is on and the clause marks it as a spare,
        replaced or missing part (see component_excluded)
    """

    def __init__(self, phrases: Sequence[str], exclude: Sequence[str] = ()):
        self.phrases = list(phrases)
        self.exclude = list(exclude)
        self._inc = _compile(self.phrases)
        self._exc = _compile(self.exclude)

    def __bool__(self) -> bool:
        return self._inc is not None

    def find(self, text: Optional[str], guard_keyword_lists: bool = False,
             context_chars: int = 40, blocked: Sequence[Tuple[int, int]] = (),
             component_guard: bool = False) -> Optional[TextMatch]:
        if not text or self._inc is None:
            return None
        t = _match_text(text)
        blocked = list(blocked)
        if self._exc:
            blocked += [m.span() for m in self._exc.finditer(t)]
        if guard_keyword_lists:
            blocked += nz.keyword_list_items(t)
            blocked += nz.compatibility_spans(t)
        for m in self._inc.finditer(t):
            s, e = m.span()
            if any(s < be and bs < e for bs, be in blocked) or negated(t, s):
                continue
            if component_guard and component_excluded(t, s, e):
                continue
            ctx = text[max(0, s - context_chars): e + context_chars]
            return TextMatch(s, e, text[s:e], ctx)
        return None


# --------------------------------------------------------------------------
# Rules and concepts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentRule:
    segment: str
    ancestor: Optional[str] = None
    domains: Optional[Tuple[str, ...]] = None
    leaf_only: bool = False

    @classmethod
    def parse(cls, spec: Any) -> "SegmentRule":
        if isinstance(spec, str):
            return cls(spec)
        if isinstance(spec, dict) and "segment" in spec:
            unknown = set(spec) - {"segment", "ancestor", "domains", "leaf_only"}
            if unknown:
                raise VocabularyError(f"unknown segment keys {unknown}")
            return cls(spec["segment"], spec.get("ancestor"),
                       tuple(spec["domains"]) if spec.get("domains") else None,
                       bool(spec.get("leaf_only", False)))
        raise VocabularyError(f"bad category segment spec: {spec!r}")

    def match(self, path: Sequence[str], domain: Optional[str]) -> bool:
        if self.domains and domain not in self.domains:
            return False
        seg = self.segment.casefold()
        anc = self.ancestor.casefold() if self.ancestor else None
        for i, name in enumerate(path):
            if name.casefold() != seg:
                continue
            if anc and anc not in (p.casefold() for p in path[:i]):
                continue
            if self.leaf_only and i != len(path) - 1:
                continue
            return True
        return False


@dataclass
class Rule:
    type: str
    rule_id: str
    confidence: float
    source: str
    evidence_type: str
    domains: Optional[Tuple[str, ...]] = None
    matcher: Optional[PhraseMatcher] = None
    field: Optional[str] = None
    segments: Tuple[SegmentRule, ...] = ()
    values: Tuple[str, ...] = ()
    contains: Tuple[str, ...] = ()
    not_contains: Tuple[str, ...] = ()
    min_number: Optional[float] = None
    domain: Optional[str] = None
    guard_keyword_lists: bool = False
    max_values: Optional[int] = None   # details_match: skip multi-valued fields
    component_guard: bool = False      # text rules: drop component mentions

    @property
    def is_text(self) -> bool:
        return self.type in _TEXT_RULES


@dataclass
class Concept:
    label: str
    node_type: NodeType
    display_name: str
    description: str
    rules: List[Rule]
    vocab_version: str
    aliases: List[str] = field(default_factory=list)
    aspect_class: Optional[str] = None
    suppress_text_in_categories: Tuple[str, ...] = ()
    review_cues: Optional["ReviewCues"] = None  # quality aspects only

    @property
    def node_id(self) -> str:
        return (interest_id if self.node_type == NodeType.INTEREST else aspect_id)(self.label)

    def node_attrs(self) -> Dict[str, Any]:
        attrs = {"label": self.label, "display_name": self.display_name,
                 "description": self.description, "vocab_version": self.vocab_version}
        if self.aliases:
            attrs["aliases"] = list(self.aliases)
        if self.aspect_class:
            attrs["aspect_class"] = self.aspect_class
        return attrs


@dataclass(frozen=True)
class ReviewCues:
    """Review-text cue phrases for one quality aspect. Used only by
    review_inference.py, on review text, never on listing text."""
    domains: Tuple[str, ...]
    positive: Tuple[str, ...]
    negative: Tuple[str, ...]
    exclude: Tuple[str, ...] = ()
    # Skip the aspect for a clause mentioning any of these words.
    context_exclude: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SuppressionClass:
    matcher: "PhraseMatcher"
    use_suggestion_exempt: bool = False


_HEAD_BREAK = re.compile(r"[,;|~(\[【]|\s[-–—]\s")


def title_head(title: str) -> str:
    """Title text before the first list separator or bracket."""
    m = _HEAD_BREAK.search(title)
    return title[:m.start()] if m else title


@dataclass
class Vocabulary:
    name: str
    version: str
    node_type: NodeType
    concepts: List[Concept]
    reserved: List[Concept] = field(default_factory=list)  # quality aspects, no rules
    # Interests only. When True, feature-text evidence can corroborate an
    # interest but never creates a MATCHES edge on its own.
    features_corroborate_only: bool = False
    # Interests only. Product classes detected from the title (party
    # supplies, nursery decor) whose text-based interest evidence is dropped.
    text_suppression: Dict[str, "SuppressionClass"] = field(default_factory=dict)
    text_domains: Tuple[str, ...] = ()
    # Aspects only: the "review_inference" block (negators, hedges,
    # confidence tiers, aggregation policy) used by review_inference.py.
    review_settings: Dict[str, Any] = field(default_factory=dict)

    def text_suppression_class(self, title: Optional[str],
                               domain: Optional[str] = None) -> Optional[str]:
        """Name of the first suppression class that names the product type.

        Only the title head (text before the first list separator or
        bracket) is checked, because later segments are SEO tails ("Beach
        Ball, Superhero Toys ... Pool Party Favors"). Classes with
        `use_suggestion_exempt` also ignore a phrase introduced by "for"/"as"
        ("Metal Coin Toys for Party Favor"). Applies only to text_domains.
        """
        if not title or (self.text_domains and domain not in self.text_domains):
            return None
        head = title_head(title)
        for name, cls in self.text_suppression.items():
            blocked = _use_suggestion_spans(head, cls.matcher) if cls.use_suggestion_exempt else ()
            if cls.matcher.find(head, blocked=blocked):
                return name
        return None

    def labels(self) -> List[str]:
        return [c.label for c in self.concepts]

    def get(self, label: str) -> Concept:
        for c in self.concepts + self.reserved:
            if c.label == label:
                return c
        raise KeyError(label)

    @property
    def version_tag(self) -> str:
        return f"{self.name}@{self.version}"


def _source_for(rule_type: str) -> Tuple[str, str]:
    return {
        "category": (SOURCE_CATEGORY_MAP, EVIDENCE_STRUCTURED),
        "details_match": (SOURCE_DETAILS, EVIDENCE_STRUCTURED),
        "details_flag": (SOURCE_DETAILS, EVIDENCE_STRUCTURED),
        "details_min_number": (SOURCE_DETAILS, EVIDENCE_STRUCTURED),
        "domain": (SOURCE_DOMAIN, EVIDENCE_STRUCTURED),
        "title": (SOURCE_TITLE_LEXICON, EVIDENCE_LISTING_STATED),
        "features": (SOURCE_FEATURE_LEXICON, EVIDENCE_LISTING_STATED),
    }[rule_type]


def _check_conf(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise VocabularyError(f"{where}: confidence must be in (0, 1], got {value!r}")
    return float(value)


def _check_label(label: Any, seen: set, where: str) -> str:
    if not isinstance(label, str) or not _LABEL_RE.match(label):
        raise VocabularyError(f"{where}: label must be lower_snake_case, got {label!r}")
    if label in seen:
        raise VocabularyError(f"{where}: duplicate label {label!r}")
    if label.replace("_", " ") in nz.RECIPIENT_OCCASION_TERMS:
        raise VocabularyError(f"{where}: {label!r} is a recipient/occasion term; "
                              "those belong to the conversation state")
    seen.add(label)
    return label


def _read(path: Path, expected: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("vocabulary") != expected:
        raise VocabularyError(f"{path}: expected vocabulary {expected!r}")
    if not re.match(r"^\d+\.\d+\.\d+$", str(data.get("version", ""))):
        raise VocabularyError(f"{path}: missing semantic version")
    return data


def load_interests(path: Path = INTERESTS_PATH) -> Vocabulary:
    """Interest entries use a compact form that compiles to generic Rules:
    category -> category rule, theme -> details_match on details.Theme,
    phrases -> title (+ features unless match_features is false) rules for
    text_domains, books_title_phrases -> a conservative Books title rule."""
    data = _read(Path(path), "interests")
    version = data["version"]
    conf = {k: _check_conf(v, f"confidence.{k}") for k, v in data["confidence"].items()}
    text_domains = tuple(data["text_domains"])
    feature_mode = data.get("feature_evidence", "standalone")
    if feature_mode not in ("standalone", "corroborating_only"):
        raise VocabularyError(f"feature_evidence must be standalone|corroborating_only, got {feature_mode!r}")
    theme_max = data.get("theme_max_values")
    suppression = {}
    for name, spec in (data.get("text_suppression") or {}).items():
        if not isinstance(spec, dict) or not spec.get("phrases"):
            raise VocabularyError(f"text_suppression.{name}: needs a non-empty 'phrases' list")
        unknown = set(spec) - {"phrases", "use_suggestion_exempt", "description"}
        if unknown:
            raise VocabularyError(f"text_suppression.{name}: unknown keys {sorted(unknown)}")
        suppression[name] = SuppressionClass(PhraseMatcher(spec["phrases"]),
                                             bool(spec.get("use_suggestion_exempt", False)))
    allowed = {"label", "display_name", "description", "aliases", "category", "theme",
               "phrases", "books_title_phrases", "match_features", "exclude",
               "suppress_text_in_categories"}
    seen: set = set()
    concepts = []
    for entry in data["interests"]:
        label = _check_label(entry.get("label"), seen, "interests")
        unknown = set(entry) - allowed
        if unknown:
            raise VocabularyError(f"{label}: unknown keys {sorted(unknown)}")
        exclude = entry.get("exclude", [])
        rules: List[Rule] = []

        def add(rtype, suffix, confidence, **kw):
            src, ev = _source_for(rtype)
            rules.append(Rule(rtype, f"{label}.{suffix}", confidence, src, ev, **kw))

        if entry.get("category"):
            add("category", "category", conf["category"],
                segments=tuple(SegmentRule.parse(s) for s in entry["category"]))
        if entry.get("theme"):
            add("details_match", "theme", conf["details_theme"], field="Theme",
                matcher=PhraseMatcher(entry["theme"]), max_values=theme_max)
        if entry.get("phrases"):
            m = PhraseMatcher(entry["phrases"], exclude)
            add("title", "title", conf["title"], domains=text_domains, matcher=m,
                guard_keyword_lists=True)
            if entry.get("match_features", True):
                add("features", "features", conf["features"], domains=text_domains,
                    matcher=m, guard_keyword_lists=True)
        if entry.get("books_title_phrases"):
            add("title", "title.books", conf["title_books"], domains=("Books",),
                matcher=PhraseMatcher(entry["books_title_phrases"], exclude),
                guard_keyword_lists=True)
        if not rules:
            raise VocabularyError(f"{label}: no rules")
        concepts.append(Concept(
            label, NodeType.INTEREST, entry.get("display_name", label),
            entry.get("description", ""), rules, version,
            aliases=list(entry.get("aliases", [])),
            suppress_text_in_categories=tuple(entry.get("suppress_text_in_categories", [])),
        ))
    return Vocabulary("interests", version, NodeType.INTEREST, concepts,
                      features_corroborate_only=feature_mode == "corroborating_only",
                      text_suppression=suppression, text_domains=text_domains)


def load_aspects(path: Path = ASPECTS_PATH) -> Vocabulary:
    data = _read(Path(path), "aspects")
    version = data["version"]
    seen: set = set()
    concepts, reserved = [], []
    for entry in data["attribute_aspects"]:
        label = _check_label(entry.get("label"), seen, "attribute_aspects")
        component_guard = bool(entry.get("component_guard", False))
        base = PhraseMatcher(entry.get("phrases", []), entry.get("exclude", []))
        rules = []
        for i, spec in enumerate(entry.get("rules", [])):
            rtype = spec.get("type")
            if rtype not in RULE_TYPES:
                raise VocabularyError(f"{label}: unknown rule type {rtype!r}")
            src, ev = _source_for(rtype)
            matcher = None
            if rtype in ("title", "features", "details_match"):
                matcher = (PhraseMatcher(spec["phrases"], entry.get("exclude", []))
                           if "phrases" in spec else base)
                if not matcher:
                    raise VocabularyError(f"{label}: {rtype} rule has no phrases")
            suffix = rtype if rtype not in ("details_match", "details_flag", "details_min_number") \
                else f"{rtype}:{spec.get('field')}"
            if rtype in _TEXT_RULES and spec.get("domains") == ["Books"]:
                suffix += ".books"
            rules.append(Rule(
                rtype, f"{label}.{suffix}", _check_conf(spec.get("confidence"), f"{label}.rules[{i}]"),
                src, ev,
                domains=tuple(spec["domains"]) if spec.get("domains") else None,
                matcher=matcher, field=spec.get("field"),
                segments=tuple(SegmentRule.parse(s) for s in spec.get("segments", [])),
                values=tuple(v.casefold() for v in spec.get("values", [])),
                contains=tuple(v.casefold() for v in spec.get("contains", [])),
                not_contains=tuple(v.casefold() for v in spec.get("not_contains", [])),
                min_number=spec.get("min"), domain=spec.get("domain"),
                component_guard=component_guard and rtype in _TEXT_RULES,
            ))
        if not rules:
            raise VocabularyError(f"{label}: no rules")
        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise VocabularyError(f"{label}: duplicate rule ids {ids}")
        concepts.append(Concept(label, NodeType.ASPECT, entry.get("display_name", label),
                                entry.get("description", ""), rules, version,
                                aspect_class=ASPECT_ATTRIBUTE))
    for entry in data.get("quality_aspects", []):
        label = _check_label(entry.get("label"), seen, "quality_aspects")
        if "rules" in entry or "phrases" in entry:
            raise VocabularyError(f"{label}: quality aspects are review-only and take no listing rules")
        reserved.append(Concept(label, NodeType.ASPECT, entry.get("display_name", label),
                                entry.get("description", ""), [], version,
                                aspect_class=ASPECT_QUALITY,
                                review_cues=_parse_review_cues(label, entry.get("review_cues"))))
    settings = data.get("review_inference") or {}
    if settings:
        _check_review_settings(settings)
    return Vocabulary("aspects", version, NodeType.ASPECT, concepts, reserved,
                      review_settings=settings)


def _parse_review_cues(label: str, spec: Any) -> Optional[ReviewCues]:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise VocabularyError(f"{label}: review_cues must be an object")
    unknown = set(spec) - {"domains", "positive", "negative", "exclude", "context_exclude"}
    if unknown:
        raise VocabularyError(f"{label}: unknown review_cues keys {sorted(unknown)}")
    pos, neg = tuple(spec.get("positive", [])), tuple(spec.get("negative", []))
    if not pos and not neg:
        raise VocabularyError(f"{label}: review_cues needs positive or negative phrases")
    overlap = {p.casefold() for p in pos} & {n.casefold() for n in neg}
    if overlap:
        raise VocabularyError(f"{label}: phrases listed as both positive and negative: {sorted(overlap)}")
    return ReviewCues(tuple(spec.get("domains", [])), pos, neg, tuple(spec.get("exclude", [])),
                      tuple(spec.get("context_exclude", [])))


def _check_review_settings(s: Dict[str, Any]) -> None:
    for k in ("direct", "negated_positive", "neutral"):
        _check_conf((s.get("confidence") or {}).get(k), f"review_inference.confidence.{k}")
    agg = s.get("aggregation") or {}
    for k in ("min_positive_reviews", "positive_to_critical_ratio",
              "concern_min_negative_reviews", "conflict_max_rating"):
        if not isinstance(agg.get(k), (int, float)) or agg[k] < 0:
            raise VocabularyError(f"review_inference.aggregation.{k} must be a non-negative number")
    if agg["min_positive_reviews"] < 2:
        raise VocabularyError("review_inference.aggregation.min_positive_reviews must be >= 2: "
                              "one review is not independent support")
