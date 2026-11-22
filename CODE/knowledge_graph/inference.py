"""Rule-based inference: Product -MATCHES-> Interest, Product -HAS_ASPECT-> Aspect.

Both extractors evaluate vocabulary Rules (see vocabulary.py) against one
corpus record and emit one edge per piece of evidence. The graph merges
evidence for the same (product, concept) into a single edge whose confidence
is the max over items (ontology.AGGREGATION_RULE).

Guarantees (each covered by tests):
  - Books: interests come from category rules and a few explicit
    title phrases. Book features and all descriptions are never used.
  - Matches inside SEO keyword lists of recipients/occasions are ignored.
  - Text matches are dropped for concepts whose suppress_text_in_categories
    appear in the product's category path (e.g. toy kitchens for cooking).
  - Interests (vocabulary v1.1+):
      * Feature text only corroborates. An interest supported by feature
        phrases alone is an incidental mention and creates no edge.
      * Products whose title marks them as party/event supplies or
        nursery/infant decor get no text- or Theme-based interests; their
        motif is decoration, not the recipient's interest. Category rules
        still apply.
      * Each MATCHES edge records match_type: "primary" when a category
        rule fired (or a Books title phrase: the book is about the topic),
        otherwise "themed" (motif, licensed character, design).
  - Feature sentences containing cross-sell wording are ignored.
  - Material aspects ignore counted kit contents and spare, replaced or
    "not included" components.
  - Only 'attribute' aspects are inferred. Quality aspects have no listing
    rules, and graph.validate() rejects any non-review edge to one.
  - Missing fields produce no evidence. They never raise.
  - MATCHES is optional. A product with no match is untouched.

Review-based quality inference and SUPPORTS live in review_inference.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from . import normalize as nz
from .extractors import Extraction
from .ontology import (
    SOURCE_CATEGORY_MAP, SOURCE_FEATURE_LEXICON, Edge, Node, Provenance, RelType,
    product_id,
)
from .vocabulary import Concept, Rule, Vocabulary

MAX_FEATURE_EVIDENCE = 2  # per concept, to keep evidence lists readable

MATCH_PRIMARY = "primary"  # the product is about / for the interest
MATCH_THEMED = "themed"    # the product carries the interest as a motif or license


@dataclass
class RecordView:
    """Normalised, read-only view of the fields the rules use."""
    asin: str
    domain: Optional[str]
    path: List[str]            # empty when categories is empty (no domain fallback)
    title: str
    features: List[str]
    details: Dict[str, Any]

    @classmethod
    def of(cls, record: Dict[str, Any]) -> "RecordView":
        path, source = nz.normalize_category_path(record)
        features = record.get("features") or []
        details = record.get("details") or {}
        return cls(
            asin=record["parent_asin"],
            domain=record.get("domain"),
            path=path if source == "categories" else [],
            title=nz.clean_text(record.get("title")) or "",
            features=[t for t in (nz.clean_text(f) for f in features if isinstance(f, str)) if t]
            if isinstance(features, list) else [],
            details=details if isinstance(details, dict) else {},
        )

    def path_has(self, segments) -> bool:
        wanted = {s.casefold() for s in segments}
        return any(p.casefold() in wanted for p in self.path)


def _prov(rule: Rule, fields, evidence, span=None, context=None) -> Provenance:
    return Provenance(source=rule.source, source_fields=fields, confidence=rule.confidence,
                      evidence=evidence, span=span, rule_id=rule.rule_id,
                      evidence_type=rule.evidence_type, context=context)


def _numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def evaluate_rule(rule: Rule, rv: RecordView) -> List[Provenance]:
    """Return the evidence this rule finds in the record (possibly none)."""
    if rule.domains and rv.domain not in rule.domains:
        return []
    t = rule.type
    if t == "category":
        if rv.path and any(s.match(rv.path, rv.domain) for s in rule.segments):
            return [_prov(rule, ["categories"], " > ".join(rv.path))]
        return []
    if t == "domain":
        return [_prov(rule, ["domain"], rv.domain)] if rv.domain == rule.domain else []
    if t.startswith("details"):
        value = nz.clean_text(rv.details.get(rule.field))
        if value is None:
            return []
        fields = [f"details.{rule.field}"]
        if t == "details_match":
            if rule.max_values and len([v for v in value.split(",") if v.strip()]) > rule.max_values:
                return []  # multi-valued lists ("Animals,Shapes,Vehicles") are motifs
            m = rule.matcher.find(value)
            return [_prov(rule, fields, value, (m.start, m.end))] if m else []
        low = value.casefold()
        if t == "details_flag":
            ok = (low in rule.values) if rule.values else any(c in low for c in rule.contains)
            if ok and not any(c in low for c in rule.not_contains):
                return [_prov(rule, fields, value)]
            return []
        if t == "details_min_number":
            nums = _numbers(value)
            return [_prov(rule, fields, value)] if nums and max(nums) >= rule.min_number else []
    if t == "title":
        m = rule.matcher.find(rv.title, rule.guard_keyword_lists,
                              component_guard=rule.component_guard)
        return [_prov(rule, ["title"], m.text, (m.start, m.end), m.context)] if m else []
    if t == "features":
        out = []
        for i, feat in enumerate(rv.features):
            blocked = nz.cross_sell_spans(feat)
            if rule.component_guard:
                blocked += nz.contents_list_items(feat)
            m = rule.matcher.find(feat, rule.guard_keyword_lists, blocked=blocked,
                                  component_guard=rule.component_guard)
            if m:
                out.append(_prov(rule, [f"features[{i}]"], m.text, (m.start, m.end), m.context))
                if len(out) >= MAX_FEATURE_EVIDENCE:
                    break
        return out
    return []


def evaluate_concept(concept: Concept, rv: RecordView,
                     suppress_non_category: bool = False) -> List[Provenance]:
    """Evidence for one concept. `suppress_non_category` drops every rule
    except category rules (used for party supplies and nursery decor)."""
    suppress_text = bool(concept.suppress_text_in_categories) and \
        rv.path_has(concept.suppress_text_in_categories)
    evidence = []
    for rule in concept.rules:
        if rule.is_text and suppress_text:
            continue
        if suppress_non_category and rule.type != "category":
            continue
        evidence.extend(evaluate_rule(rule, rv))
    return evidence


def incidental_only(evidence: List[Provenance]) -> bool:
    """True when every item is feature text: an incidental mention."""
    return bool(evidence) and all(p.source == SOURCE_FEATURE_LEXICON for p in evidence)


def match_type(evidence: List[Provenance], domain: Optional[str]) -> str:
    if any(p.source == SOURCE_CATEGORY_MAP for p in evidence) or domain == "Books":
        return MATCH_PRIMARY
    return MATCH_THEMED


def make_concept_extractor(vocab: Vocabulary, rel: RelType) -> Callable[[Dict[str, Any]], Extraction]:
    """Build an extractor for extractors.extract_record()."""
    flag = "interest_matches" if rel == RelType.MATCHES else "aspect_matches"

    is_interest = rel == RelType.MATCHES

    def extract(record: Dict[str, Any]) -> Extraction:
        rv = RecordView.of(record)
        ex = Extraction()
        pid = product_id(rv.asin)
        suppression = vocab.text_suppression_class(rv.title, rv.domain) if is_interest else None
        if suppression:
            ex.flags[f"interest_text_suppressed={suppression}"] = True
        n, incidental = 0, 0
        for concept in vocab.concepts:
            evidence = evaluate_concept(concept, rv, suppress_non_category=bool(suppression))
            if not evidence:
                continue
            if is_interest and vocab.features_corroborate_only and incidental_only(evidence):
                incidental += 1
                continue
            n += 1
            attrs = {"vocab_version": vocab.version_tag}
            if is_interest:
                attrs["match_type"] = match_type(evidence, rv.domain)
            ex.nodes.append(Node(concept.node_id, concept.node_type, concept.node_attrs()))
            for prov in evidence:
                ex.edges.append(Edge(pid, rel, concept.node_id, prov, dict(attrs)))
        ex.flags[flag] = n
        ex.flags[f"{flag}_none"] = n == 0
        if is_interest:
            ex.flags["interest_incidental_mentions"] = incidental
        return ex

    extract.__name__ = f"extract_{vocab.name}"
    return extract


def make_interest_extractor(vocab: Vocabulary):
    return make_concept_extractor(vocab, RelType.MATCHES)


def make_aspect_extractor(vocab: Vocabulary):
    return make_concept_extractor(vocab, RelType.HAS_ASPECT)
