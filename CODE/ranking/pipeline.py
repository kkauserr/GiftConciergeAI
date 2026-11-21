import argparse
import json
import time
from dataclasses import dataclass, field

from CODE.ranking.filters import apply_filters
from CODE.ranking.fusion import normalize_results, rrf
from CODE.ranking.ranker import DEFAULT_WEIGHTS, rank
from CODE.retrieval.bm25_baseline import BM25Retriever, Catalog, TfidfRetriever, keyword_query


@dataclass
class RankConfig:
    lexical: str = "bm25"
    use_fusion: bool = True
    use_filters: bool = True
    use_ranking: bool = True
    use_mmr: bool = True
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    pool_size: int = 200
    top_k: int = 10
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    min_relevance: float = 0.25


class GiftRanker:
    def __init__(self, index_dir, load_tfidf=True):
        self.catalog = Catalog.load(index_dir)
        self.retrievers = {"bm25": BM25Retriever.load(index_dir)}
        if load_tfidf:
            self.retrievers["tfidf"] = TfidfRetriever.load(index_dir)

    def recommend(self, constraints, semantic_results=None, config=None):
        cfg = config or RankConfig()
        t0 = time.perf_counter()
        query = keyword_query(constraints)
        lexical = self.retrievers[cfg.lexical].search(query, cfg.pool_size)
        semantic = normalize_results(semantic_results)[: cfg.pool_size]

        if cfg.use_fusion and semantic:
            candidates, source_ranks = rrf({cfg.lexical: lexical, "semantic": semantic}, k=cfg.rrf_k)
        else:
            candidates = lexical
            source_ranks = {a: {f"{cfg.lexical}_rank": i} for i, (a, _) in enumerate(lexical, 1)}

        dropped = {}
        if cfg.use_filters:
            candidates, dropped = apply_filters(candidates, self.catalog, constraints)

        if cfg.use_ranking:
            weights = cfg.weights
        else:
            weights = {"relevance": 1.0}
        results = rank(candidates, self.catalog, constraints, weights=weights, top_k=cfg.top_k,
                       use_mmr=cfg.use_mmr and cfg.use_ranking, mmr_lambda=cfg.mmr_lambda,
                       min_relevance=cfg.min_relevance if cfg.use_ranking else 0.0)

        items = []
        for row in results.to_dict("records"):
            items.append({
                "rank": int(row["rank"]),
                "parent_asin": row["parent_asin"],
                "title": row["title"],
                "domain": row["domain"],
                "price": float(row["price_num"]),
                "average_rating": float(row["average_rating"]),
                "rating_number": int(row["rating_number"]),
                "final_score": round(float(row["final_score"]), 4),
                "signals": {k: round(float(row[k]), 4) for k in ["relevance", "quality", "popularity", "budget_fit"]},
                "sources": source_ranks.get(row["parent_asin"], {}),
            })

        return {
            "query": query,
            "constraints": constraints,
            "n_candidates": len(lexical) + len(semantic),
            "n_after_filters": len(candidates),
            "dropped": dropped,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": items,
        }


def main():
    p = argparse.ArgumentParser(description="Run filtering + ranking for one gift request")
    p.add_argument("--index", default="indexes/sample")
    p.add_argument("--constraints", required=True, help="JSON string or path to a .json file")
    p.add_argument("--semantic", default=None, help="optional JSON file: [[parent_asin, score], ...]")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--no-mmr", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    c = args.constraints
    constraints = json.load(open(c)) if c.endswith(".json") else json.loads(c)
    semantic = json.load(open(args.semantic)) if args.semantic else None

    ranker = GiftRanker(args.index)
    out = ranker.recommend(constraints, semantic, RankConfig(top_k=args.k, use_mmr=not args.no_mmr))
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"query: {out['query']!r} | after filters: {out['n_after_filters']} | dropped: {out['dropped']} | {out['latency_ms']} ms")
    for r in out["results"]:
        s = r["signals"]
        print(f"{r['rank']:>2}. {r['final_score']:.3f}  ${r['price']:<7.2f} {r['average_rating']}*({r['rating_number']:>5}) "
              f"[{r['domain'][:8]}] {r['title'][:60]}  rel={s['relevance']:.2f} q={s['quality']:.2f} "
              f"pop={s['popularity']:.2f} fit={s['budget_fit']:.2f}")


if __name__ == "__main__":
    main()
