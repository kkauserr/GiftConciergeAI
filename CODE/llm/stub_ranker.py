"""PLACEHOLDER ranker until Disha's BM25/RRF/filtering/ranking stage is ready.

Same interface the real stage should expose:

    rank(retrieved, constraints, products, top_k=5) -> [{"parent_asin", "score", "rank"}]

    retrieved   : semantic results [{"parent_asin", "score", "rank"}]
    constraints : GiftConstraints.to_handoff() dict
    products    : {parent_asin: product record}

What the stub does (deliberately simple):
  * hard filters: price_num within [budget_min, budget_max]; domain equals
    category_preference; no exclusion term in title/categories; product age
    range overlaps target_age_range when both are known
  * score = semantic score + RATING_WEIGHT * (average_rating - 3)
  * no relaxation: if filters leave fewer than top_k, it returns fewer
"""

RATING_WEIGHT = 0.02


def _excluded(product, exclusions):
    if not exclusions:
        return False
    haystack = " ".join([product.get("title") or "", product.get("domain") or ""]
                        + list(product.get("categories") or [])).lower()
    for term in exclusions:
        term = term.lower()
        stem = term[:-1] if term.endswith("s") and len(term) > 3 else term
        if stem in haystack:
            return True
    return False


def _age_ok(product, target):
    if not target:
        return True
    lo, hi = product.get("age_min"), product.get("age_max")
    if lo is None and hi is None:
        return True
    t_lo, t_hi = target
    if hi is not None and t_lo is not None and t_lo > hi:
        return False
    if lo is not None and t_hi is not None and t_hi < lo:
        return False
    return True


def passes_filters(product, constraints):
    price = product.get("price_num")
    if constraints.get("budget_min") is not None and (price is None or price < constraints["budget_min"]):
        return False
    if constraints.get("budget_max") is not None and (price is None or price > constraints["budget_max"]):
        return False
    category = constraints.get("category_preference")
    if category and product.get("domain") != category:
        return False
    if _excluded(product, constraints.get("exclusions")):
        return False
    return _age_ok(product, constraints.get("target_age_range"))


def rank(retrieved, constraints, products, top_k=5):
    scored = []
    for hit in retrieved:
        product = products.get(hit["parent_asin"])
        if product is None or not passes_filters(product, constraints):
            continue
        rating = product.get("average_rating") or 3.0
        scored.append((hit["score"] + RATING_WEIGHT * (rating - 3.0), hit["parent_asin"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"parent_asin": asin, "score": round(score, 6), "rank": i + 1}
            for i, (score, asin) in enumerate(scored[:top_k])]
