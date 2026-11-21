import argparse
import copy
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

from CODE.ranking.filters import check_product
from CODE.ranking.pipeline import GiftRanker, RankConfig
from CODE.ranking.ranker import item_similarity


def build_configs(has_semantic):
    full = RankConfig()
    configs = {
        "bm25_only": RankConfig(use_fusion=False, use_filters=False, use_ranking=False),
        "tfidf_only": RankConfig(lexical="tfidf", use_fusion=False, use_filters=False, use_ranking=False),
        "bm25_filtered": RankConfig(use_fusion=False, use_ranking=False),
        "full": full,
    }
    for sig in ["relevance", "quality", "popularity", "budget_fit"]:
        c = copy.deepcopy(full)
        c.weights[sig] = 0.0
        configs[f"no_{sig}"] = c
    c = copy.deepcopy(full); c.use_mmr = False; configs["no_mmr"] = c
    c = copy.deepcopy(full); c.min_relevance = 0.0; configs["no_relevance_gate"] = c
    if has_semantic:
        c = copy.deepcopy(full); c.use_fusion = False; configs["no_fusion"] = c
    return configs


def dcg(grades):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades))


def label_metrics(asins, labels, k):
    grades = [labels.get(a, 0) for a in asins[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    return {
        f"p@{k}": sum(g > 0 for g in grades) / k,
        f"ndcg@{k}": dcg(grades) / dcg(ideal) if ideal and dcg(ideal) > 0 else 0.0,
    }


def list_metrics(out, ranker, constraints, k):
    items = out["results"]
    n = len(items)
    m = {"n_results": n, "latency_ms": out["latency_ms"]}
    if n == 0:
        return m
    asins = [r["parent_asin"] for r in items]
    meta = ranker.catalog.get(asins).reindex(asins)
    ok = [not check_product(row, constraints) for row in meta.to_dict("records")]
    m["constraint_satisfaction"] = float(np.mean(ok))
    m["domain_coverage"] = meta["domain"].nunique() / min(3, n)
    m["category_diversity"] = meta["specific_category"].nunique() / n
    if n > 1:
        sim = item_similarity(meta)
        m["intra_list_similarity"] = float(np.mean([sim[i, j] for i, j in combinations(range(n), 2)]))
    m["mean_rating"] = float(meta["average_rating"].mean())
    m["median_rating_count"] = float(meta["rating_number"].median())
    if constraints.get("budget_max"):
        m["mean_budget_use"] = float((meta["price_num"] / constraints["budget_max"]).mean())
    return m


def main():
    p = argparse.ArgumentParser(description="Ablation study for the classical ranking pipeline")
    p.add_argument("--index", default="indexes/sample")
    p.add_argument("--queries", default="CODE/ranking/dev_queries.json")
    p.add_argument("--semantic-file", default=None, help="JSON {qid: [[parent_asin, score], ...]}")
    p.add_argument("--semantic-standin", action="store_true",
                   help="use TF-IDF as a stand-in second list to test fusion (not real semantic retrieval)")
    p.add_argument("--labels", default=None, help="CSV with qid,parent_asin,grade")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--out", default="EVALUATIONS/results/ranking")
    args = p.parse_args()

    queries = json.load(open(args.queries))
    ranker = GiftRanker(args.index)
    semantic = json.load(open(args.semantic_file)) if args.semantic_file else {}
    labels = {}
    if args.labels:
        lab = pd.read_csv(args.labels).dropna(subset=["grade"])
        for qid, g in lab.groupby("qid"):
            labels[qid] = dict(zip(g["parent_asin"].astype(str), g["grade"].astype(int)))

    has_semantic = bool(semantic) or args.semantic_standin
    configs = build_configs(has_semantic)
    rows, runs = [], {}

    for q in queries:
        qid, cons = q["qid"], q["constraints"]
        if args.semantic_standin:
            from CODE.retrieval.bm25_baseline import keyword_query
            sem = ranker.retrievers["tfidf"].search(keyword_query(cons), 200)
        else:
            sem = semantic.get(qid)
        full_asins = None
        for name, cfg in configs.items():
            cfg = copy.deepcopy(cfg)
            cfg.top_k = args.k
            out = ranker.recommend(cons, sem, cfg)
            asins = [r["parent_asin"] for r in out["results"]]
            runs.setdefault(name, {})[qid] = out["results"]
            m = {"config": name, "qid": qid, **list_metrics(out, ranker, cons, args.k)}
            if name == "full":
                full_asins = set(asins)
            if qid in labels:
                m.update(label_metrics(asins, labels[qid], args.k))
            rows.append((m, asins))
        for m, asins in rows:
            if m["qid"] == qid:
                m["overlap_with_full"] = len(set(asins) & full_asins) / max(len(full_asins), 1)

    per_query = pd.DataFrame([m for m, _ in rows])
    summary = per_query.drop(columns=["qid"]).groupby("config", sort=False).mean(numeric_only=True).round(3)

    os.makedirs(args.out, exist_ok=True)
    per_query.to_csv(os.path.join(args.out, "ablation_per_query.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "ablation_summary.csv"))
    with open(os.path.join(args.out, "ablation_runs.json"), "w") as f:
        json.dump(runs, f, indent=1)
    with open(os.path.join(args.out, "ablation_summary.md"), "w") as f:
        f.write(summary.to_markdown())

    pd.set_option("display.width", 250)
    print(summary.to_string())
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
