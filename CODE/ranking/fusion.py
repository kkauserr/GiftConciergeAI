def normalize_results(results):
    out = []
    for r in results or []:
        if isinstance(r, dict):
            out.append((str(r["parent_asin"]), float(r.get("score", 0.0))))
        else:
            out.append((str(r[0]), float(r[1])))
    return out


def rrf(ranked_lists, k=60, weights=None):
    weights = weights or {}
    fused, ranks = {}, {}
    for name, results in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, (asin, _) in enumerate(normalize_results(results), 1):
            fused[asin] = fused.get(asin, 0.0) + w / (k + rank)
            ranks.setdefault(asin, {})[f"{name}_rank"] = rank
    order = sorted(fused.items(), key=lambda x: -x[1])
    return order, ranks
