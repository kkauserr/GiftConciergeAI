"""Turn a GiftConcierge.handle() result into the response the frontend renders.

The frontend contract is documented in CODE/frontend/README.md. This module
only reshapes and enriches data that the pipeline already produced:

  * product details (title, price, rating, category) from the corpus record
  * image_url from the image lookup, amazon_url from parent_asin
  * why_it_fits = the grounded explanation written by the LLM layer
  * one human-readable highlight taken from the evidence the explanation cited
    (a quoted review, or a knowledge-graph claim), never an evidence ID

Internal fields (scores, retrieval sources, evidence IDs) stay out of the
response unless GC_DEBUG=1, where they go into a separate `debug` block.
"""

import re

REVIEW_PREFIX = re.compile(r"^(\d+(?:\.\d+)?/5 review|Review):\s*")
MAX_HIGHLIGHT_CHARS = 220


def _clip(text, limit=MAX_HIGHLIGHT_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _highlight(explanation):
    """The single most useful cited evidence line, in plain language."""
    used = explanation.get("used_evidence") or []
    for item in used:
        if item.get("field") == "review_samples" and item.get("text"):
            return {"text": _clip(REVIEW_PREFIX.sub("", item["text"])), "source": "review"}
    for item in used:
        text = item.get("text") or ""
        if item.get("field") == "kg" and text and not text.lower().startswith("caution"):
            return {"text": _clip(text), "source": "insight"}
    return None


def _number(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _recommendation(item, product, image_url):
    asin = item["parent_asin"]
    highlight = _highlight(item)
    rating_count = product.get("rating_number")
    return {
        "parent_asin": asin,
        "rank": item.get("rank"),
        "title": product.get("title") or "Untitled product",
        "price": _number(product.get("price_num", product.get("price"))),
        "rating": _number(product.get("average_rating")),
        "rating_count": int(rating_count) if isinstance(rating_count, (int, float)) else None,
        "category": product.get("domain"),
        "image_url": image_url,
        "amazon_url": f"https://www.amazon.com/dp/{asin}",
        "why_it_fits": item.get("explanation") or "",
        "review_evidence": [highlight] if highlight else [],
    }


def _results_message(count, turn_had_changes):
    if turn_had_changes:
        return "Here's an updated set based on what you just told me."
    noun = "gift" if count == 1 else "gifts"
    return f"I found {count} {noun} worth considering. These are the strongest matches based on what you told me."


def to_ui_response(raw, engine, debug=False, previous_results=False):
    """raw: GiftConcierge.handle() output. engine: Engine (products + images)."""
    response = {
        "type": raw["type"],
        "turn": raw.get("turn"),
        "slot": raw.get("slot"),
        "constraints": raw.get("constraints") or {},
        "suggestions": [],
    }

    if raw["type"] == "question":
        response["message"] = raw["message"]
    else:
        recommendations = []
        for item in raw.get("explanations") or []:
            product = engine.products.get(item["parent_asin"])
            if product is None:
                continue
            recommendations.append(_recommendation(item, product, engine.image_for(item["parent_asin"])))
        response["recommendations"] = recommendations
        response["message"] = _results_message(len(recommendations), previous_results)

    if debug:
        response["debug"] = {
            "state_changes": raw.get("state_changes"),
            "conflicts": raw.get("conflicts"),
            "extraction_error": raw.get("extraction_error"),
            "query_text": raw.get("query_text"),
            "ranked": raw.get("ranked"),
            "explanations": [
                {k: e.get(k) for k in ("parent_asin", "evidence_ids", "invalid_citations", "fallback")}
                for e in raw.get("explanations") or []
            ],
        }
    return response
