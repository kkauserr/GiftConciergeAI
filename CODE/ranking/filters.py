import re

import pandas as pd

DOMAIN_ALIASES = {
    "book": "Books", "books": "Books", "novel": "Books", "novels": "Books",
    "toy": "Toys & Games", "toys": "Toys & Games", "game": "Toys & Games", "games": "Toys & Games",
    "handmade": "Handmade Products", "handmade products": "Handmade Products",
}


def _exclusion_pattern(term):
    words = term.lower().split()
    if words[-1].endswith("s") and not words[-1].endswith("ss") and len(words[-1]) > 3:
        words[-1] = words[-1][:-1]
    words = [re.escape(w) for w in words]
    return re.compile(r"\b" + r"\s+".join(words) + r"(s|es)?\b")


def check_product(row, constraints):
    reasons = []

    price = row.get("price_num")
    bmax, bmin = constraints.get("budget_max"), constraints.get("budget_min")
    if bmax is not None or bmin is not None:
        if price is None or pd.isna(price):
            reasons.append("no_price")
        else:
            if bmax is not None and price > bmax:
                reasons.append("over_budget")
            if bmin is not None and price < bmin:
                reasons.append("under_budget")

    age = constraints.get("recipient_age")
    if age is not None:
        amin, amax = row.get("age_min"), row.get("age_max")
        if amin is not None and not pd.isna(amin) and age < amin:
            reasons.append("too_young")
        if amax is not None and not pd.isna(amax) and age > amax:
            reasons.append("too_old")

    text = " ".join(str(row.get(c) or "") for c in ("title", "categories", "specific_category")).lower()
    for term in constraints.get("exclusions") or []:
        term = term.strip().lower()
        if not term:
            continue
        if DOMAIN_ALIASES.get(term) == row.get("domain") or _exclusion_pattern(term).search(text):
            reasons.append(f"excluded:{term}")

    return reasons


def apply_filters(candidates, catalog, constraints):
    meta = catalog.get([a for a, _ in candidates])
    kept, dropped = [], {}
    for asin, score in candidates:
        if asin not in meta.index:
            dropped["missing_metadata"] = dropped.get("missing_metadata", 0) + 1
            continue
        reasons = check_product(meta.loc[asin].to_dict(), constraints)
        if reasons:
            for r in reasons:
                key = r.split(":")[0]
                dropped[key] = dropped.get(key, 0) + 1
        else:
            kept.append((asin, score))
    return kept, dropped
