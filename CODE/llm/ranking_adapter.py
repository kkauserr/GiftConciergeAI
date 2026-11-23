"""Plug Disha's GiftRanker (CODE/ranking) into the conversation loop.

    from CODE.ranking.pipeline import GiftRanker      # repo root on sys.path
    GiftConcierge(..., ranker=HybridRanker(GiftRanker("indexes/sample")))

GiftRanker does BM25 + RRF fusion + budget/age/exclusion filters + ranking + MMR.
It doesn't know two of our constraints, so they are checked here on its output:
category_preference, and target_age_range when only an age group is known (its
age filter needs an exact recipient_age). MMR works on a pool of 50, so that many
are requested and then cut to top_k.
"""

from llm.stub_ranker import _age_ok

RANK_POOL = 50


class HybridRanker:
    def __init__(self, gift_ranker, config=None):
        self.gift_ranker = gift_ranker
        self.config = config
        self.last_output = None   # full recommend() output: signals, dropped, latency

    def __call__(self, retrieved, constraints, products, top_k=5):
        if self.config is None:
            from CODE.ranking.pipeline import RankConfig
            self.config = RankConfig(top_k=RANK_POOL)
        out = self.gift_ranker.recommend(constraints, semantic_results=retrieved,
                                         config=self.config)
        self.last_output = out
        category = constraints.get("category_preference")
        kept = []
        for item in out["results"]:
            product = products.get(item["parent_asin"])
            if product is None or (category and item["domain"] != category):
                continue
            if not _age_ok(product, constraints.get("target_age_range")):
                continue
            kept.append({"parent_asin": item["parent_asin"], "score": item["final_score"],
                         "rank": len(kept) + 1, "signals": item["signals"],
                         "sources": item["sources"]})
            if len(kept) == top_k:
                break
        return kept
