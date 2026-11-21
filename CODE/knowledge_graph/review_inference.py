"""Review evidence: Review -SUPPORTS-> quality Aspect, and review-derived
Product -HAS_ASPECT-> quality Aspect.

Two layers, so the rules can later be swapped for a trained classifier:

1. A ReviewAspectClassifier turns one review's text into AspectMentions
   (aspect label, item polarity, cue span, evidence clause). The default
   RuleBasedReviewClassifier uses the `review_cues` of the 12 quality
   aspects in vocab/aspects.json. Any object with `version` and
   `classify(text, domain)` can replace it.

2. make_review_aspect_extractor() runs a classifier over every unique
   review of a record (same IDs and de-duplication as extract_reviews) and
   emits:
     - one SUPPORTS edge per (review, aspect) with any mention. Each
       evidence item keeps the clause, cue span, item polarity and
       confidence; the edge keeps the review-level `polarity`
       (positive | negative | neutral | mixed), rating, verified_purchase
       and helpful_vote.
     - Product -HAS_ASPECT-> aspect only when the aggregation policy is
       met: enough distinct positive reviews and few negative or mixed
       ones. Negative evidence never becomes HAS_ASPECT; aspects with a
       negative majority are recorded as concerns in the product's
       `review_aspect_signals`.

Rules (each covered by tests):
  - Only review text is read. Seller title/features/description never are.
  - A star rating is never evidence by itself. A positive mention in a
    review rated <= conflict_max_rating is kept but flagged rating_conflict
    and does not count toward HAS_ASPECT.
  - A negated positive cue ("not sturdy") is negative. A negated negative
    cue ("didn't break") is neutral, not positive. Hedged or questioning
    sentences ("hope it lasts", "is it durable?") are neutral.
  - Sentences are split into clauses at contrast words, so "cute but broke
    in a week" is judged per clause.
  - A review with positive and negative items for one aspect is "mixed";
    mixed reviews count against, never for, an aspect.
  - Reviews with identical text count once (text_hash), as do exact
    duplicate reviews (same Review ID).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .extractors import Extraction, unique_reviews
from .ontology import (
    EVIDENCE_REVIEW, SOURCE_REVIEW_AGGREGATE, SOURCE_REVIEW_TEXT, Edge, Polarity,
    Provenance, RelType, product_id,
)
from .vocabulary import Concept, Vocabulary, _compile, _match_text, load_aspects

POS, NEG, NEU, MIX = (Polarity.POSITIVE.value, Polarity.NEGATIVE.value,
                      Polarity.NEUTRAL.value, Polarity.MIXED.value)

STATUS_SUPPORTED = "supported"      # HAS_ASPECT created
STATUS_CONCERN = "concern"          # negative majority; never HAS_ASPECT
STATUS_INSUFFICIENT = "insufficient"
STATUS_NEUTRAL_ONLY = "neutral_only"


# --------------------------------------------------------------------------
# Classifier interface
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AspectMention:
    label: str
    polarity: str               # item polarity: positive | negative | neutral
    confidence: float           # heuristic score, not a probability
    cue: str                    # matched cue text
    span: Tuple[int, int]       # cue offsets in the review text
    clause: str                 # clause containing the cue (the evidence)
    sentence: str               # full sentence (context)
    reason: str                 # direct | negated_positive | negated_negative | hedged | question
    cue_polarity: str           # polarity of the cue phrase before negation


class ReviewAspectClassifier(Protocol):
    version: str

    def classify(self, text: str, domain: Optional[str]) -> List[AspectMention]:
        ...


# --------------------------------------------------------------------------
# Rule-based classifier
# --------------------------------------------------------------------------

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass
class _CompiledCues:
    concept: Concept
    domains: Tuple[str, ...]
    positive: Optional[re.Pattern]
    negative: Optional[re.Pattern]
    exclude: Optional[re.Pattern]
    context_exclude: Optional[re.Pattern]


class RuleBasedReviewClassifier:
    """Phrase cues + clause-level negation, hedging and contrast handling."""

    def __init__(self, vocab: Optional[Vocabulary] = None):
        vocab = vocab or load_aspects()
        s = vocab.review_settings
        if not s:
            raise ValueError("aspect vocabulary has no review_inference settings")
        self.version = f"{s.get('classifier', 'review_rules')}@{vocab.version}"
        conf = s["confidence"]
        self.conf_direct = conf["direct"]
        self.conf_negated = conf["negated_positive"]
        self.conf_neutral = conf["neutral"]
        self.negators = {n.lower() for n in s.get("negators", [])}
        self.window = int(s.get("negation_window_tokens", 3))
        self.exceptions = [e.lower() for e in s.get("negation_exceptions", [])]
        hedges = s.get("hedges", [])
        self.hedge_re = _compile(hedges) if hedges else None
        contrast = s.get("contrast_markers", [])
        self.contrast_re = _compile(contrast) if contrast else None
        self.cues: List[_CompiledCues] = []
        for c in vocab.reserved:
            if c.review_cues is None:
                continue
            rc = c.review_cues
            self.cues.append(_CompiledCues(c, rc.domains, _compile(rc.positive),
                                           _compile(rc.negative), _compile(rc.exclude),
                                           _compile(rc.context_exclude)))

    # -- text segmentation ------------------------------------------------
    def _clauses(self, text: str, s: int, e: int) -> List[Tuple[int, int]]:
        """Split text[s:e] at contrast markers and semicolons."""
        cuts = [s]
        if self.contrast_re:
            cuts += [m.start() + s for m in self.contrast_re.finditer(text[s:e])]
        cuts += [m.end() + s for m in re.finditer(r";", text[s:e])]
        cuts = sorted(set(c for c in cuts if s <= c < e))
        return [(a, b) for a, b in zip(cuts, cuts[1:] + [e]) if text[a:b].strip()]

    # -- polarity modifiers ------------------------------------------------
    @staticmethod
    def _scope(text: str, clause_start: int, start: int) -> str:
        """Clause text before the cue, cut at the nearest comma/colon/dash."""
        before = text[clause_start:start]
        for sep in (",", ":", "(", " - ", "--"):
            if sep in before:
                before = before[before.rindex(sep) + len(sep):]
        return before

    def _negated(self, text: str, clause_start: int, start: int) -> bool:
        before = self._scope(text, clause_start, start).lower()
        for exc in self.exceptions:
            before = before.replace(exc, " ")
        tokens = _TOKEN.findall(before)[-self.window:]
        return any(t in self.negators for t in tokens)

    def _hedged(self, text: str, clause_start: int, start: int) -> bool:
        return bool(self.hedge_re and self.hedge_re.search(self._scope(text, clause_start, start)))

    # -- main ----------------------------------------------------------------
    def classify(self, text: str, domain: Optional[str]) -> List[AspectMention]:
        if not text:
            return []
        t = _match_text(text)
        cues = [c for c in self.cues if not c.domains or domain in c.domains]
        out: List[AspectMention] = []
        for sm in _SENTENCE.finditer(t):
            ss, se = sm.span()
            sentence = text[ss:se].strip()
            question = t[ss:se].rstrip().endswith("?")
            for cs, ce in self._clauses(t, ss, se):
                clause = text[cs:ce].strip(" ,")
                for cc in cues:
                    out.extend(self._clause_mentions(text, t, cs, ce, cc, clause, sentence, question))
        return out

    def _clause_mentions(self, text, t, cs, ce, cc, clause, sentence, question):
        seg = t[cs:ce]
        hits = []
        for pol, pat in ((POS, cc.positive), (NEG, cc.negative)):
            if pat:
                hits += [(m.start() + cs, m.end() + cs, pol) for m in pat.finditer(seg)]
        if not hits:
            return []
        if cc.context_exclude and cc.context_exclude.search(seg):
            return []
        if cc.exclude:
            blocked = [(m.start() + cs, m.end() + cs) for m in cc.exclude.finditer(seg)]
            hits = [h for h in hits if not any(h[0] < be and bs < h[1] for bs, be in blocked)]
        # A cue inside a longer cue of the same aspect yields to it:
        # "worth the money" inside "not worth the money".
        hits = [h for h in hits if not any(o != h and o[0] <= h[0] and h[1] <= o[1]
                                           and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]
        mentions, seen = [], set()
        for s, e, cue_pol in sorted(hits):
            if question:
                pol, reason, conf = NEU, "question", self.conf_neutral
            elif self._hedged(t, cs, s):
                pol, reason, conf = NEU, "hedged", self.conf_neutral
            elif self._negated(t, cs, s):
                if cue_pol == POS:
                    pol, reason, conf = NEG, "negated_positive", self.conf_negated
                else:
                    pol, reason, conf = NEU, "negated_negative", self.conf_neutral
            else:
                pol, reason, conf = cue_pol, "direct", self.conf_direct
            if pol in seen:          # one item per polarity per clause
                continue
            seen.add(pol)
            mentions.append(AspectMention(cc.concept.label, pol, conf, text[s:e], (s, e),
                                          clause, sentence, reason, cue_pol))
        return mentions


# --------------------------------------------------------------------------
# Aggregation policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregationPolicy:
    min_positive_reviews: int = 2
    positive_to_critical_ratio: float = 2.0
    concern_min_negative_reviews: int = 2
    conflict_max_rating: float = 2.0

    @classmethod
    def from_vocab(cls, vocab: Vocabulary) -> "AggregationPolicy":
        a = (vocab.review_settings or {}).get("aggregation") or {}
        return cls(int(a.get("min_positive_reviews", 2)),
                   float(a.get("positive_to_critical_ratio", 2.0)),
                   int(a.get("concern_min_negative_reviews", 2)),
                   float(a.get("conflict_max_rating", 2.0)))

    def as_dict(self) -> Dict[str, Any]:
        return {"min_positive_reviews": self.min_positive_reviews,
                "positive_to_critical_ratio": self.positive_to_critical_ratio,
                "concern_min_negative_reviews": self.concern_min_negative_reviews,
                "conflict_max_rating": self.conflict_max_rating}

    def status(self, pos: int, neg: int, mixed: int, neutral: int) -> str:
        if pos >= self.min_positive_reviews and pos >= self.positive_to_critical_ratio * (neg + mixed):
            return STATUS_SUPPORTED
        if neg >= self.concern_min_negative_reviews and neg > pos:
            return STATUS_CONCERN
        if pos + neg + mixed == 0:
            return STATUS_NEUTRAL_ONLY
        return STATUS_INSUFFICIENT


def review_polarity_of_items(polarities) -> str:
    """Review-level polarity for one aspect from its item polarities:
    all positive (ignoring neutral) -> positive, all negative -> negative,
    both -> mixed, none -> neutral."""
    pols = set(polarities) - {NEU}
    if pols == {POS}:
        return POS
    if pols == {NEG}:
        return NEG
    if pols == {POS, NEG}:
        return MIX
    return NEU


def review_polarity(mentions: Sequence[AspectMention]) -> str:
    return review_polarity_of_items(m.polarity for m in mentions)


def _rating(review: Dict[str, Any]) -> Optional[float]:
    r = review.get("rating")
    return float(r) if isinstance(r, (int, float)) else None


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------

def make_review_aspect_extractor(
        vocab: Optional[Vocabulary] = None,
        classifier: Optional[ReviewAspectClassifier] = None,
        policy: Optional[AggregationPolicy] = None) -> Callable[[Dict[str, Any]], Extraction]:
    """Build an extractor for extractors.extract_record(). Must run after
    extract_reviews in the same extractor list, which creates the Review
    nodes these edges start from."""
    vocab = vocab or load_aspects()
    classifier = classifier or RuleBasedReviewClassifier(vocab)
    policy = policy or AggregationPolicy.from_vocab(vocab)
    concepts = {c.label: c for c in vocab.reserved}

    def extract(record: Dict[str, Any]) -> Extraction:
        ex = Extraction()
        if "review_samples" not in record:
            return ex
        asin, domain = record["parent_asin"], record.get("domain")
        unique, _ = unique_reviews(record)
        counts = defaultdict(int)
        # label -> list of (UniqueReview, review polarity, mentions, rating_conflict)
        per_label: Dict[str, List[tuple]] = defaultdict(list)
        for u in unique:
            text = u.review.get("text") or ""
            by_label: Dict[str, List[AspectMention]] = defaultdict(list)
            for m in classifier.classify(text, domain):
                if m.label in concepts:
                    by_label[m.label].append(m)
            rating = _rating(u.review)
            for label, mentions in by_label.items():
                pol = review_polarity(mentions)
                conflict = pol == POS and rating is not None and rating <= policy.conflict_max_rating
                per_label[label].append((u, pol, mentions, conflict))
                counts[f"review_supports_{pol}"] += 1
                attrs = {"polarity": pol, "parent_asin": asin, "rating": rating,
                         "verified_purchase": u.review.get("verified_purchase"),
                         "helpful_vote": u.review.get("helpful_vote"),
                         "rating_conflict": conflict, "classifier": classifier.version,
                         "vocab_version": vocab.version_tag}
                for m in mentions:
                    counts[f"review_mentions_{m.polarity}"] += 1
                    ex.edges.append(Edge(u.id, RelType.SUPPORTS, concepts[label].node_id, Provenance(
                        source=SOURCE_REVIEW_TEXT, source_fields=["review_samples.text"],
                        confidence=m.confidence, evidence=m.clause, span=m.span,
                        # rule_id: "<aspect>.review.<cue polarity>:<reason>", e.g.
                        # "durability.review.positive:negated_positive"
                        rule_id=f"{label}.review.{m.cue_polarity}:{m.reason}",
                        evidence_type=EVIDENCE_REVIEW, context=m.sentence,
                        evidence_polarity=m.polarity, review_id=u.id,
                    ), dict(attrs)))
        signals = {}
        for label, results in per_label.items():
            signals[label] = _aggregate(ex, asin, concepts[label], results, policy, vocab,
                                        classifier, len(unique))
            counts[f"review_aspect_{signals[label]['status']}"] += 1
        if signals:
            ex.product_attrs["review_aspect_signals"] = signals
        ex.flags.update(counts)
        return ex

    extract.__name__ = "extract_review_aspects"
    return extract


def _aggregate(ex: Extraction, asin: str, concept: Concept, results: List[tuple],
               policy: AggregationPolicy, vocab: Vocabulary, classifier, n_reviews: int) -> Dict[str, Any]:
    """Count independent reviews per polarity and, if the policy is met,
    emit Product -HAS_ASPECT-> aspect with one evidence item per positive
    review."""
    seen_text = set()
    pos, neg, mixed, neutral, conflicts, pos_verified = [], 0, 0, 0, 0, 0
    for u, pol, mentions, conflict in results:
        if u.text_hash in seen_text:     # identical text counts once
            continue
        seen_text.add(u.text_hash)
        if pol == POS and conflict:
            conflicts += 1
        elif pol == POS:
            pos.append((u, mentions))
            pos_verified += bool(u.review.get("verified_purchase"))
        elif pol == NEG:
            neg += 1
        elif pol == MIX:
            mixed += 1
        else:
            neutral += 1
    status = policy.status(len(pos), neg, mixed, neutral)
    signal = {"positive": len(pos), "negative": neg, "mixed": mixed, "neutral": neutral,
              "rating_conflict": conflicts, "status": status}
    if status != STATUS_SUPPORTED:
        return signal
    attrs = {"polarity": POS, "n_positive": len(pos), "n_negative": neg, "n_mixed": mixed,
             "n_neutral": neutral, "n_rating_conflict": conflicts, "n_reviews": n_reviews,
             "n_positive_verified": pos_verified, "aggregation_policy": policy.as_dict(),
             "classifier": classifier.version, "vocab_version": vocab.version_tag}
    for u, mentions in pos:
        best = max((m for m in mentions if m.polarity == POS), key=lambda m: m.confidence)
        ex.edges.append(Edge(product_id(asin), RelType.HAS_ASPECT, concept.node_id, Provenance(
            source=SOURCE_REVIEW_AGGREGATE, source_fields=["review_samples.text"],
            confidence=best.confidence, evidence=best.clause, span=best.span,
            rule_id=f"{concept.label}.review_aggregate", evidence_type=EVIDENCE_REVIEW,
            context=best.sentence, evidence_polarity=POS, review_id=u.id,
        ), dict(attrs)))
    return signal
