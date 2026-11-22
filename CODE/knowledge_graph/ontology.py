"""Gift Concierge KG ontology: entity types, relationship types, IDs, provenance.

Six entity types (Product, Brand, Category, Interest, Aspect, Review) and the
relationships between them. SUBCATEGORY_OF is the one addition beyond the
original proposal, used for category hierarchy traversal.

This module has no third-party dependencies so it can be reused by a streaming
full-corpus build that never materialises an in-memory graph.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence


class NodeType(str, Enum):
    PRODUCT = "Product"
    BRAND = "Brand"
    CATEGORY = "Category"
    INTEREST = "Interest"
    ASPECT = "Aspect"
    REVIEW = "Review"


class RelType(str, Enum):
    MADE_BY = "MADE_BY"                # Product  -> Brand
    BELONGS_TO = "BELONGS_TO"          # Product  -> Category (leaf of path)
    SUBCATEGORY_OF = "SUBCATEGORY_OF"  # Category -> parent Category
    MATCHES = "MATCHES"                # Product  -> Interest   (inferred)
    HAS_ASPECT = "HAS_ASPECT"          # Product  -> Aspect     (inferred: listing or reviews)
    ABOUT = "ABOUT"                    # Review   -> Product
    SUPPORTS = "SUPPORTS"              # Review   -> Aspect     (inferred, carries polarity)


# Allowed (source type, target type) per relationship. Used for validation.
EDGE_SCHEMA: Dict[RelType, tuple] = {
    RelType.MADE_BY: (NodeType.PRODUCT, NodeType.BRAND),
    RelType.BELONGS_TO: (NodeType.PRODUCT, NodeType.CATEGORY),
    RelType.SUBCATEGORY_OF: (NodeType.CATEGORY, NodeType.CATEGORY),
    RelType.MATCHES: (NodeType.PRODUCT, NodeType.INTEREST),
    RelType.HAS_ASPECT: (NodeType.PRODUCT, NodeType.ASPECT),
    RelType.ABOUT: (NodeType.REVIEW, NodeType.PRODUCT),
    RelType.SUPPORTS: (NodeType.REVIEW, NodeType.ASPECT),
}


class Polarity(str, Enum):
    """Evidence direction for SUPPORTS edges.

    A negative review about durability is NEGATIVE evidence for the durability
    aspect, never support for "this product is durable". Despite the edge
    name, a SUPPORTS edge only says the review is evidence about the aspect;
    its polarity says in which direction.

    Each evidence item (one sentence or clause) is positive, negative or
    neutral. A review whose items for one aspect disagree is MIXED at the
    edge level, and mixed reviews never count as support.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# Polarities allowed on a single evidence item ("mixed" is edge-level only).
ITEM_POLARITIES = frozenset({Polarity.POSITIVE.value, Polarity.NEGATIVE.value,
                             Polarity.NEUTRAL.value})


# Provenance "source" values.
SOURCE_METADATA = "metadata"                    # copied directly from a corpus field
SOURCE_METADATA_FALLBACK = "metadata_fallback"  # derived from a secondary metadata field
SOURCE_CATEGORY_MAP = "category_map"            # curated category-path rule
SOURCE_DETAILS = "details"                      # curated rule over a structured details field
SOURCE_DOMAIN = "domain"                        # rule over the corpus domain
SOURCE_TITLE_LEXICON = "title_lexicon"          # vocabulary phrase found in title
SOURCE_FEATURE_LEXICON = "feature_lexicon"      # vocabulary phrase found in features
SOURCE_REVIEW_TEXT = "review_text"              # cue phrase found in one review's text
SOURCE_REVIEW_AGGREGATE = "review_aggregate"    # several independent reviews agree

# Evidence types for HAS_ASPECT.
EVIDENCE_STRUCTURED = "structured"          # details / categories / domain
EVIDENCE_LISTING_STATED = "listing_stated"  # seller-written title or feature text
EVIDENCE_REVIEW = "review"                  # review text (SUPPORTS, quality HAS_ASPECT)

# Aspect classes. Quality aspects may only be supported by reviews.
ASPECT_ATTRIBUTE = "attribute"
ASPECT_QUALITY = "quality"

# Confidence aggregation for an edge with several evidence items.
# Confidences are heuristic rule scores, not calibrated probabilities. Evidence
# for the same edge is often correlated (a title and a category that restate
# each other), so scores are not combined. The edge takes the max.
AGGREGATION_RULE = "max"


@dataclass(frozen=True)
class Provenance:
    """Where an edge came from. Required on every edge, direct or inferred.

    source        - extraction method (see SOURCE_* constants).
    source_fields - corpus fields the edge was derived from.
    confidence    - heuristic rule score in [0, 1]. 1.0 for direct metadata.
                    Not a calibrated probability.
    evidence      - raw text/value supporting the edge.
    span          - optional (start, end) character offsets of evidence
                    within the source field, for text matches.
    rule_id       - vocabulary rule that fired, e.g. "dogs.title".
    evidence_type - for HAS_ASPECT: structured | listing_stated | review.
    context       - short window of surrounding text, for human review and
                    LLM grounding. Not part of evidence identity.
    evidence_polarity - review evidence only: positive | negative | neutral
                    for this item (see Polarity). The edge-level direction
                    is the edge attribute `polarity`, which may be "mixed".
    review_id     - review evidence only: the Review node the item came from.
                    `span` then indexes that review's text.
    """
    source: str
    source_fields: Sequence[str]
    confidence: float = 1.0
    evidence: Optional[str] = None
    span: Optional[Sequence[int]] = None
    rule_id: Optional[str] = None
    evidence_type: Optional[str] = None
    context: Optional[str] = None
    evidence_polarity: Optional[str] = None
    review_id: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.evidence_polarity is not None and self.evidence_polarity not in ITEM_POLARITIES:
            raise ValueError(f"evidence_polarity must be one of {sorted(ITEM_POLARITIES)}, got {self.evidence_polarity!r}")
        object.__setattr__(self, "source_fields", tuple(self.source_fields))
        if self.span is not None:
            object.__setattr__(self, "span", tuple(self.span))


@dataclass
class Node:
    id: str
    type: NodeType
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type.value, **self.attrs}


@dataclass
class Edge:
    src: str
    rel: RelType
    dst: str
    provenance: Provenance
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prov = asdict(self.provenance)
        prov["source_fields"] = list(prov["source_fields"])
        if prov["span"] is not None:
            prov["span"] = list(prov["span"])
        return {"src": self.src, "rel": self.rel.value, "dst": self.dst,
                **prov, **self.attrs}


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Stable, case/accent-insensitive slug. "Rubie's" -> "rubies", "&" -> "and"."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ").replace("'", "").replace("’", "")
    return _SLUG_STRIP.sub("-", text).strip("-")


def product_id(parent_asin: str) -> str:
    return f"product:{parent_asin}"


def brand_id(name: str) -> str:
    return f"brand:{slugify(name)}"


def category_id(path: Sequence[str]) -> str:
    """Path-based ID: category names are not unique across parents
    (e.g. "Literature & Fiction" appears under four different parents)."""
    if not path:
        raise ValueError("category path must be non-empty")
    return "category:" + "/".join(slugify(p) for p in path)


_VOCAB_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _vocab_label(label: str) -> str:
    """Vocabulary labels are used unchanged in IDs, so they stay identical to
    the labels shared with the conversation component (no slug rewriting)."""
    if not _VOCAB_LABEL.match(label):
        raise ValueError(f"vocabulary label must be lower_snake_case: {label!r}")
    return label


def interest_id(label: str) -> str:
    return f"interest:{_vocab_label(label)}"


def aspect_id(label: str) -> str:
    return f"aspect:{_vocab_label(label)}"


def normalize_for_hash(text: Optional[str]) -> str:
    """Whitespace- and case-insensitive form used for content fingerprints."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


# Fields that identify a review, in order. A field is used only if present
# and non-empty in the record, so the ID strategy works both for the current
# final corpus (no user/review IDs) and for richer raw records that do carry
# user_id / asin. Rating and helpful_vote are deliberately excluded: they are
# mutable or noisy and should not change a review's identity.
REVIEW_IDENTITY_FIELDS: Sequence[str] = (
    "review_id", "user_id", "asin", "timestamp", "title", "text",
)


def review_fingerprint(parent_asin: str, review: Dict[str, Any],
                       identity_fields: Iterable[str] = REVIEW_IDENTITY_FIELDS) -> str:
    """SHA-1 over parent_asin + whichever identity fields are present.

    Each component is written as "name=value" so absent fields cannot be
    confused with empty ones and field order is fixed.
    """
    parts = [f"parent_asin={parent_asin}"]
    for name in identity_fields:
        value = review.get(name)
        if value is None or value == "":
            continue
        if name in ("title", "text"):
            value = normalize_for_hash(value)
        parts.append(f"{name}={value}")
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def review_id(parent_asin: str, review: Dict[str, Any],
              identity_fields: Iterable[str] = REVIEW_IDENTITY_FIELDS) -> str:
    """Deterministic review node ID. Exact duplicates (same product, same
    identity fields after normalisation) collapse to the same ID."""
    return f"review:{parent_asin}:{review_fingerprint(parent_asin, review, identity_fields)[:16]}"


def text_hash(text: Optional[str]) -> str:
    """Content-only hash for detecting near-duplicates (same text, different
    timestamp or product). Stored as an attribute, not used as identity."""
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()[:16]
