"""Evidence-grounded (RAG) explanations for the final ranked products.

Each product's evidence is turned into short, ID-tagged lines:

    P1.title          product title
    P1.price          price
    P1.rating         Amazon average rating + count
    P1.category       most specific category
    P1.age            recommended age range (if known)
    P1.brand          brand / author
    P1.feature[j]     product feature strings
    P1.description[0] first description line (only when there are no features)
    P1.review_stats   review_count / average_review_rating / verified count
    P1.review[j]      review samples (most helpful first)
    P1.kg[j]          knowledge-graph facts (pluggable, see below)

The LLM must return, per product, a short explanation plus the evidence IDs it
used. We then check those IDs against what was actually in the prompt: unknown
IDs or IDs from another product are dropped and reported in
`invalid_citations`. The `used_evidence` field (ID, field and exact text) is
what faithfulness evaluation should score the explanation against.

KG facts: pass `kg_facts` as either {parent_asin: [fact, ...]} or any object
with get_facts(parent_asin) -> [fact, ...]. A fact can be a string or a dict
like {"subject": ..., "relation": ..., "object": ...} (or {"text": ...}).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

log = logging.getLogger(__name__)

MAX_FEATURES = 4
MAX_REVIEWS = 3
MAX_KG_FACTS = 5
MAX_CHARS = 280


class KGFactProvider(Protocol):
    def get_facts(self, parent_asin: str) -> list: ...


@dataclass
class Evidence:
    id: str
    parent_asin: str
    field: str
    text: str
    path_id: Optional[str] = None   # KG reasoning path (kg evidence only)


@dataclass
class Explanation:
    parent_asin: str
    rank: int
    explanation: str
    evidence_ids: list
    used_evidence: list
    invalid_citations: list = field(default_factory=list)
    fallback: bool = False

    def to_dict(self):
        return asdict(self)


def _clip(text, limit=MAX_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _fact_text(fact):
    if isinstance(fact, str):
        return fact
    if isinstance(fact, dict):
        if "text" in fact:
            return str(fact["text"])
        parts = [fact.get(k) for k in ("subject", "relation", "object")]
        if all(parts):
            return " ".join(str(p) for p in parts)
    return json.dumps(fact, default=str)


def get_kg_facts(kg_facts, parent_asin):
    if kg_facts is None:
        return []
    if hasattr(kg_facts, "get_facts"):
        return list(kg_facts.get_facts(parent_asin) or [])
    return list(kg_facts.get(parent_asin, []) or [])


def collect_evidence(product, pid, kg=None):
    """Return the ID-tagged evidence list for one product (pid like 'P1')."""
    asin = product["parent_asin"]
    ev = []

    def add(suffix, field_name, text, path_id=None):
        if text is not None and str(text).strip():
            ev.append(Evidence(f"{pid}.{suffix}", asin, field_name, _clip(text), path_id))

    add("title", "title", product.get("title"))
    price = product.get("price_num", product.get("price"))
    if price is not None:
        add("price", "price_num", f"Price: ${float(price):,.2f}")
    if product.get("average_rating") is not None:
        count = product.get("rating_number")
        suffix = f" from {count:,} ratings" if count else ""
        add("rating", "average_rating",
            f"Average rating {product['average_rating']}/5{suffix}")
    category = product.get("specific_category") or " > ".join(product.get("categories") or [])
    add("category", "specific_category", category and f"Category: {category}")
    lo, hi = product.get("age_min"), product.get("age_max")
    if lo is not None or hi is not None:
        if lo is not None and hi is not None:
            ages = f"{lo:g}-{hi:g} years"
        elif lo is not None:
            ages = f"{lo:g}+ years"
        else:
            ages = f"up to {hi:g} years"
        add("age", "age_min/age_max", f"Recommended age: {ages}")
    add("brand", "brand_or_author",
        product.get("brand_or_author") and f"By {product['brand_or_author']}")

    features = [f for f in product.get("features") or [] if str(f).strip()]
    for j, feat in enumerate(features[:MAX_FEATURES]):
        add(f"feature[{j}]", "features", feat)
    if not features:
        desc = [d for d in product.get("description") or [] if str(d).strip()]
        if desc:
            add("description[0]", "description", desc[0])

    if product.get("review_count"):
        stats = f"{product['review_count']:,} reviews"
        if product.get("average_review_rating") is not None:
            stats += f", average review rating {product['average_review_rating']:.1f}/5"
        if product.get("verified_review_count"):
            stats += f", {product['verified_review_count']:,} verified purchases"
        add("review_stats", "review_count/average_review_rating", stats)
    reviews = sorted(product.get("review_samples") or [],
                     key=lambda r: r.get("helpful_vote") or 0, reverse=True)
    for j, review in enumerate(reviews[:MAX_REVIEWS]):
        body = " - ".join(x for x in (review.get("title"), review.get("text")) if x)
        if not body:
            continue
        stars = f"{review['rating']:g}/5 review: " if review.get("rating") is not None else "Review: "
        add(f"review[{j}]", "review_samples", stars + body)

    for j, fact in enumerate(get_kg_facts(kg, asin)[:MAX_KG_FACTS]):
        add(f"kg[{j}]", "kg", _fact_text(fact),
            fact.get("path_id") if isinstance(fact, dict) else None)
    return ev


EXPLANATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["explanations"],
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["product_id", "explanation", "evidence_ids"],
                "properties": {
                    "product_id": {"type": "string"},
                    "explanation": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

EXPLANATION_SYSTEM_PROMPT = """\
You explain why each recommended product could be a good gift, for a gift \
recommender. Use ONLY the evidence lines given for that product. Every claim \
must be supported by an evidence line, and evidence_ids must list every line \
you relied on (use the exact IDs, e.g. "P1.feature[0]"). Never use evidence from \
a different product. Do not invent features, prices, ratings, reviews or facts, \
and do not use outside knowledge about the product. If the evidence only weakly \
matches the shopper's request, say so plainly instead of overstating the fit. \
Keep the wording of knowledge-graph (kg) lines: "theme" means themed, not made for \
that hobby; "the listing says" stays the seller's claim; review counts stay as given. \
A "Caution:" line is a drawback, never praise. \
Write 1-2 sentences (max 50 words) per product, addressed to the shopper. Do not \
put evidence IDs in the explanation text. Return one entry per product ID."""


def _constraints_summary(constraints):
    data = constraints.to_handoff() if hasattr(constraints, "to_handoff") else dict(constraints)
    if "exclusions_stated" in data:   # the normalized list is only for the filter
        data["exclusions"] = data.pop("exclusions_stated")
    for key in ("needs_clarification", "clarification_question", "interest_labels"):
        data.pop(key, None)
    return {k: v for k, v in data.items() if v not in (None, [], False)}


def build_explanation_prompt(constraints, product_ids, evidence_by_pid):
    blocks = []
    for pid in product_ids:
        lines = "\n".join(f"[{e.id}] {e.text}" for e in evidence_by_pid[pid])
        blocks.append(f"Product {pid}:\n{lines}")
    user = ("Shopper's request (constraints):\n"
            + json.dumps(_constraints_summary(constraints), indent=2)
            + "\n\nRecommended products and their evidence:\n\n" + "\n\n".join(blocks))
    return [{"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _fallback(pid, evidence):
    by_suffix = {e.id.split(".", 1)[1]: e for e in evidence}
    used = [by_suffix[k] for k in ("title", "price", "rating") if k in by_suffix]
    title = by_suffix["title"].text if "title" in by_suffix else "This product"
    text = title
    if "price" in by_suffix:
        text += f" ({by_suffix['price'].text.replace('Price: ', '')})"
    if "rating" in by_suffix:
        text += f", {by_suffix['rating'].text[0].lower()}{by_suffix['rating'].text[1:]}"
    return text + ".", used


def explain_recommendations(llm, ranked, products, constraints, kg_facts=None, top_k=5):
    """Explain the top_k ranked products using only their evidence.

    ranked: [{"parent_asin", "rank", ...}] from the ranking stage.
    products: {parent_asin: product record}.
    Returns {"explanations": [Explanation.to_dict()...], "evidence": {pid: [...]}}.
    """
    chosen = [r for r in sorted(ranked, key=lambda r: r["rank"]) if r["parent_asin"] in products]
    chosen = chosen[:top_k]
    pids = [f"P{i}" for i in range(1, len(chosen) + 1)]
    evidence_by_pid = {pid: collect_evidence(products[r["parent_asin"]], pid, kg_facts)
                       for pid, r in zip(pids, chosen)}

    raw_items = {}
    if chosen:
        try:
            raw = llm.complete_json(build_explanation_prompt(constraints, pids, evidence_by_pid),
                                    EXPLANATION_SCHEMA, "gift_explanations")
            for item in raw.get("explanations", []):
                if item.get("product_id") in evidence_by_pid and item["product_id"] not in raw_items:
                    raw_items[item["product_id"]] = item
        except Exception as exc:  # provider error -> template fallback
            log.warning("explanation generation failed: %s", exc)

    results = []
    for pid, r in zip(pids, chosen):
        evidence = evidence_by_pid[pid]
        known = {e.id: e for e in evidence}
        item = raw_items.get(pid)
        text = (item or {}).get("explanation", "").strip()
        if item and text:
            cited = list(dict.fromkeys(item.get("evidence_ids", [])))
            used = [known[i] for i in cited if i in known]
            invalid = [i for i in cited if i not in known]
            fallback = False
        else:
            text, used = _fallback(pid, evidence)
            invalid, fallback = [], True
        results.append(Explanation(
            parent_asin=r["parent_asin"],
            rank=r["rank"],
            explanation=text,
            evidence_ids=[e.id for e in used],
            used_evidence=[asdict(e) for e in used],
            invalid_citations=invalid,
            fallback=fallback,
        ))
    return {
        "explanations": [x.to_dict() for x in results],
        "evidence": {pid: [asdict(e) for e in evs] for pid, evs in evidence_by_pid.items()},
    }
