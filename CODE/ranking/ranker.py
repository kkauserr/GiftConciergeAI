import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIGNALS = ["relevance", "quality", "popularity", "budget_fit"]

DEFAULT_WEIGHTS = {"relevance": 0.7, "quality": 0.1, "popularity": 0.0, "budget_fit": 0.2}

BUDGET_TARGET = 0.7
BUDGET_WIDTH = 0.25


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def bayesian_rating(avg, count, prior_mean, prior_count):
    avg = np.nan_to_num(np.asarray(avg, dtype=float), nan=prior_mean)
    count = np.nan_to_num(np.asarray(count, dtype=float), nan=0.0)
    return (count * avg + prior_count * prior_mean) / (count + prior_count)


def budget_fit(prices, constraints):
    bmax, bmin = constraints.get("budget_max"), constraints.get("budget_min")
    prices = np.asarray(prices, dtype=float)
    if bmax is None:
        return np.ones_like(prices)
    if bmin is not None and bmin < bmax:
        target = (bmin + bmax) / 2 / bmax
    else:
        target = BUDGET_TARGET
    ratio = prices / bmax
    return np.exp(-((ratio - target) ** 2) / (2 * BUDGET_WIDTH ** 2))


def compute_signals(candidates, catalog, constraints):
    asins = [a for a, _ in candidates]
    meta = catalog.get(asins).reindex(asins)
    stats = catalog.stats

    rel = minmax([s for _, s in candidates])
    bayes = bayesian_rating(meta["average_rating"], meta["rating_number"],
                            stats["mean_rating"], stats["prior_count"])
    quality = np.clip((bayes - 1.0) / 4.0, 0, 1)
    pop = np.clip(np.log1p(meta["rating_number"].fillna(0).astype(float)) / stats["pop_cap"], 0, 1)
    fit = budget_fit(meta["price_num"], constraints)

    df = pd.DataFrame({
        "parent_asin": asins,
        "retrieval_score": [s for _, s in candidates],
        "relevance": rel,
        "quality": quality,
        "popularity": pop.values,
        "budget_fit": fit,
        "bayes_rating": bayes,
    })
    return df, meta


def weighted_score(df, weights):
    w = {k: float(weights.get(k, 0.0)) for k in SIGNALS}
    total = sum(w.values())
    if total <= 0:
        return df["relevance"].values
    return sum(w[k] * df[k].values for k in SIGNALS) / total


def item_similarity(meta):
    text = (meta["title"].fillna("") + " " + meta["specific_category"].fillna("") + " "
            + meta["categories"].fillna("")).tolist()
    try:
        vecs = TfidfVectorizer(stop_words="english").fit_transform(text)
        cos = cosine_similarity(vecs)
    except ValueError:
        cos = np.zeros((len(text), len(text)))
    cat = meta["specific_category"].fillna("").to_numpy(dtype=object)
    dom = meta["domain"].fillna("").to_numpy(dtype=object)
    same_cat = (cat[:, None] == cat[None, :]) & (cat[:, None] != "")
    same_dom = dom[:, None] == dom[None, :]
    return 0.6 * cos + 0.25 * same_cat + 0.15 * same_dom


def mmr(scores, sim, k, lam=0.7):
    n = len(scores)
    k = min(k, n)
    selected, remaining = [], list(range(n))
    max_sim = np.zeros(n)
    while remaining and len(selected) < k:
        vals = lam * scores[remaining] - (1 - lam) * max_sim[remaining]
        best = remaining[int(np.argmax(vals))]
        selected.append(best)
        remaining.remove(best)
        max_sim = np.maximum(max_sim, sim[best])
    return selected


def rank(candidates, catalog, constraints, weights=None, top_k=10, use_mmr=True,
         mmr_lambda=0.9, mmr_pool=50, min_relevance=0.25):
    if not candidates:
        return pd.DataFrame()
    weights = DEFAULT_WEIGHTS if weights is None else weights
    df, meta = compute_signals(candidates, catalog, constraints)

    # relevance gate: quality/popularity can't lift an item that barely matches the request
    keep = df["relevance"].values >= min_relevance
    if not keep.any():
        keep = df["relevance"].values == df["relevance"].max()
    df, meta = df[keep].reset_index(drop=True), meta[keep]
    df["final_score"] = weighted_score(df, weights)
    order = np.argsort(-df["final_score"].values, kind="stable")

    if use_mmr:
        pool = order[:mmr_pool]
        sim = item_similarity(meta.iloc[pool])
        picked = [pool[i] for i in mmr(df["final_score"].values[pool], sim, top_k, mmr_lambda)]
    else:
        picked = list(order[:top_k])

    out = df.iloc[picked].copy()
    m = meta.iloc[picked]
    for col in ["title", "domain", "price_num", "average_rating", "rating_number", "specific_category"]:
        out[col] = m[col].values
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)
