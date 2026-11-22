"""Direct extractors: turn one corpus record into KG nodes and edges.

Extractors are pure functions (record -> Extraction) with no graph state, so
the same code can run in-memory on the 900-product sample or stream over the
full corpus. Duplicate nodes across records (shared brands and categories)
are merged by the graph layer.

Implemented here (all directly derivable from verified fields):
    Product, Brand, Category nodes
    MADE_BY, BELONGS_TO, SUBCATEGORY_OF edges
    Review nodes + ABOUT edges, built from `review_samples` when that field
    exists (records without it produce no reviews).

Inferred relationships live elsewhere: MATCHES and listing-based HAS_ASPECT
in inference.py, SUPPORTS and review-derived HAS_ASPECT in review_inference.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import normalize as nz
from .ontology import (
    SOURCE_METADATA, SOURCE_METADATA_FALLBACK, Edge, Node, NodeType,
    Provenance, RelType, brand_id, category_id, product_id, review_id,
    text_hash,
)

# Product attributes copied straight from metadata. Text fields
# (features/description/bm25_text/...) stay in the corpus and are looked up
# by parent_asin when evidence is needed, which keeps the graph small.
PRODUCT_ATTR_FIELDS = (
    "parent_asin", "domain", "title", "price_num", "average_rating",
    "rating_number", "store", "main_category", "specific_category",
    "age_min", "age_max",
)

# Review aggregate fields, present only in the final (review-joined) corpus.
REVIEW_STAT_FIELDS = (
    "review_count", "rated_review_count", "average_review_rating",
    "verified_review_count", "helpful_votes",
)

REVIEW_ATTR_FIELDS = (
    "title", "text", "rating", "helpful_vote", "verified_purchase", "timestamp",
)


@dataclass
class Extraction:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    # Data-quality observations for reporting, e.g. {"brand_status": "placeholder"}
    flags: Dict[str, Any] = field(default_factory=dict)
    # Extra attributes for this record's Product node (e.g. review signals).
    # extract_record() merges them into the Product node it created.
    product_attrs: Dict[str, Any] = field(default_factory=dict)

    def extend(self, other: "Extraction") -> None:
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        self.flags.update(other.flags)
        self.product_attrs.update(other.product_attrs)


class RecordError(ValueError):
    """Raised for records that cannot become a Product node."""


def extract_product(record: Dict[str, Any]) -> Extraction:
    asin = nz.clean_text(record.get("parent_asin"))
    if not asin:
        raise RecordError("record has no parent_asin")
    attrs = {f: record.get(f) for f in PRODUCT_ATTR_FIELDS}
    attrs["parent_asin"] = asin
    attrs["age_range_valid"] = nz.age_range_valid(record.get("age_min"), record.get("age_max"))
    attrs["has_review_fields"] = "review_samples" in record
    for f in REVIEW_STAT_FIELDS:
        if f in record:
            attrs[f] = record[f]
    ex = Extraction(nodes=[Node(product_id(asin), NodeType.PRODUCT, attrs)])
    if attrs["age_range_valid"] is False:
        ex.flags["age_range_invalid"] = True
    if nz.has_encoding_issue(record.get("title")):
        ex.flags["title_encoding_issue"] = True
    return ex


def extract_brand(record: Dict[str, Any]) -> Extraction:
    """Brand node + Product -MADE_BY-> Brand. No edge for placeholders or missing."""
    asin = record["parent_asin"]
    name, source_field, status = nz.resolve_brand(record)
    ex = Extraction(flags={"brand_status": status})
    if name is None:
        return ex
    kind = nz.brand_kind(record.get("domain"), source_field)
    bid = brand_id(name)
    ex.nodes.append(Node(bid, NodeType.BRAND, {"name": name, "kinds": [kind]}))
    raw = (record.get("brand_or_author") if source_field == "brand_or_author"
           else (record.get("details") or {}).get(source_field.split(".", 1)[1]))
    ex.edges.append(Edge(
        product_id(asin), RelType.MADE_BY, bid,
        Provenance(
            source=SOURCE_METADATA if source_field == "brand_or_author" else SOURCE_METADATA_FALLBACK,
            source_fields=[source_field], confidence=1.0, evidence=raw,
        ),
        {"kind": kind},
    ))
    return ex


def extract_categories(record: Dict[str, Any]) -> Extraction:
    """One Category node per path prefix, SUBCATEGORY_OF child->parent,
    and Product -BELONGS_TO-> leaf category."""
    asin = record["parent_asin"]
    path, source_field = nz.normalize_category_path(record)
    ex = Extraction(flags={"category_source": source_field})
    if not path:
        return ex
    domain = record.get("domain")
    ids = [category_id(path[: i + 1]) for i in range(len(path))]
    for i, (cid, name) in enumerate(zip(ids, path)):
        ex.nodes.append(Node(cid, NodeType.CATEGORY, {
            "name": name, "depth": i, "path": path[: i + 1],
            "domains": [domain] if domain else [],
            "encoding_issue": nz.has_encoding_issue(name),
        }))
        if i > 0:
            ex.edges.append(Edge(
                cid, RelType.SUBCATEGORY_OF, ids[i - 1],
                Provenance(SOURCE_METADATA, [source_field], 1.0),
            ))
    ex.edges.append(Edge(
        product_id(asin), RelType.BELONGS_TO, ids[-1],
        Provenance(
            source=SOURCE_METADATA if source_field == "categories" else SOURCE_METADATA_FALLBACK,
            source_fields=[source_field], confidence=1.0, evidence=" > ".join(path),
        ),
    ))
    if source_field == "domain":
        ex.flags["category_fallback"] = True
    return ex


def extract_reviews(record: Dict[str, Any]) -> Extraction:
    """Review nodes + Review -ABOUT-> Product from `review_samples`.

    - Field absent: returns an empty extraction flagged
      reviews_available=False. No records are fabricated.
    - Reviews with empty text are skipped; they carry no evidence.
    - Exact duplicates within a product collapse to one node (same
      deterministic ID) with duplicate_count incremented.
    - text_hash is stored so near-duplicates (same text, different
      timestamp) can be analysed later without merging them silently.
    """
    asin = record["parent_asin"]
    if "review_samples" not in record:
        return Extraction(flags={"reviews_available": False})
    unique, skipped = unique_reviews(record)
    ex = Extraction(flags={"reviews_available": True, "reviews_skipped": skipped,
                           "reviews_duplicate": sum(u.count - 1 for u in unique)})
    for u in unique:
        review = u.review
        attrs = {f: review.get(f) for f in REVIEW_ATTR_FIELDS}
        attrs.update(parent_asin=asin, text_hash=u.text_hash, duplicate_count=u.count)
        for optional in ("user_id", "asin", "review_id"):
            if review.get(optional) not in (None, ""):
                attrs[optional] = review[optional]
        ex.nodes.append(Node(u.id, NodeType.REVIEW, attrs))
        ex.edges.append(Edge(
            u.id, RelType.ABOUT, product_id(asin),
            Provenance(SOURCE_METADATA, ["review_samples", "parent_asin"], 1.0),
        ))
    return ex


@dataclass
class UniqueReview:
    id: str                 # deterministic Review node ID (ontology.review_id)
    review: Dict[str, Any]  # first occurrence in review_samples
    text_hash: str
    count: int = 1          # exact duplicates merged into this one


def unique_reviews(record: Dict[str, Any]) -> tuple:
    """(unique reviews in file order, number skipped) for one record.

    Single source of the review identity and de-duplication policy, shared
    by extract_reviews and review_inference so SUPPORTS edges always point
    at the Review nodes that were built. Non-dict entries and reviews with
    empty text are skipped; exact duplicates (same ID) are merged.
    """
    asin = record["parent_asin"]
    by_id: Dict[str, UniqueReview] = {}
    skipped = 0
    for review in record.get("review_samples") or []:
        if not isinstance(review, dict) or not nz.clean_text(review.get("text")):
            skipped += 1
            continue
        rid = review_id(asin, review)
        if rid in by_id:
            by_id[rid].count += 1
            continue
        by_id[rid] = UniqueReview(rid, review, text_hash(review.get("text")))
    return list(by_id.values()), skipped


DEFAULT_EXTRACTORS: List[Callable[[Dict[str, Any]], Extraction]] = [
    extract_brand, extract_categories, extract_reviews,
]


def extract_record(record: Dict[str, Any],
                   extractors: Optional[Iterable[Callable]] = None) -> Extraction:
    """Run the product extractor, then each registered extractor. Later
    inference extractors (interests, aspects) plug in via `extractors`."""
    ex = extract_product(record)
    product = ex.nodes[0]
    rec = dict(record, parent_asin=product.attrs["parent_asin"])
    for fn in (DEFAULT_EXTRACTORS if extractors is None else extractors):
        ex.extend(fn(rec))
    product.attrs.update(ex.product_attrs)
    return ex
