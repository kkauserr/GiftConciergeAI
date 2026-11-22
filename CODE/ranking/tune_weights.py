import argparse
import copy
import itertools
import json
import os

import numpy as np
import pandas as pd

from CODE.ranking.ablation import build_configs, label_metrics
from CODE.ranking.pipeline import GiftRanker, RankConfig
from CODE.ranking.ranker import SIGNALS


def export_pool(ranker, queries, k, out_csv):
    rows = []
    for q in queries:
        seen = set()
        for name, cfg in build_configs(False).items():
            cfg = copy.deepcopy(cfg)
            cfg.top_k = k
            for r in ranker.recommend(q["constraints"], None, cfg)["results"]:
                if r["parent_asin"] in seen:
                    continue
                seen.add(r["parent_asin"])
                rows.append({"qid": q["qid"], "request": q.get("request", ""), "parent_asin": r["parent_asin"],
                             "title": r["title"], "domain": r["domain"], "price": r["price"], "grade": ""})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} (query, product) pairs to {out_csv}")
    print("Fill the grade column: 2 = great gift for this request, 1 = okay, 0 = not relevant")


def weight_grid(step):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for combo in itertools.product(vals, repeat=len(SIGNALS) - 1):
        rest = round(1 - sum(combo), 2)
        if rest < 0 or combo[0] == 0:
            continue
        yield dict(zip(SIGNALS, list(combo) + [rest]))


def tune(ranker, queries, labels, k, step, lambdas):
    queries = [q for q in queries if q["qid"] in labels]
    if not queries:
        raise SystemExit("No labelled queries found. Run the 'pool' step and fill in grades first.")
    results = []
    for weights in weight_grid(step):
        for lam in lambdas:
            cfg = RankConfig(weights=weights, mmr_lambda=lam, top_k=k)
            scores = []
            for q in queries:
                out = ranker.recommend(q["constraints"], None, cfg)
                asins = [r["parent_asin"] for r in out["results"]]
                scores.append(label_metrics(asins, labels[q["qid"]], k)[f"ndcg@{k}"])
            results.append({**weights, "mmr_lambda": lam, f"ndcg@{k}": float(np.mean(scores))})
    df = pd.DataFrame(results).sort_values(f"ndcg@{k}", ascending=False)
    return df


def main():
    p = argparse.ArgumentParser(description="Tune ranking weights on labelled development queries")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["pool", "tune"]:
        s = sub.add_parser(name)
        s.add_argument("--index", default="indexes/sample")
        s.add_argument("--queries", default="CODE/ranking/dev_queries.json")
        s.add_argument("-k", type=int, default=10)
    sub.choices["pool"].add_argument("--out", default="CODE/ranking/dev_labels.csv")
    t = sub.choices["tune"]
    t.add_argument("--labels", default="CODE/ranking/dev_labels.csv")
    t.add_argument("--step", type=float, default=0.1)
    t.add_argument("--lambdas", default="0.5,0.7,0.9")
    t.add_argument("--out", default="EVALUATIONS/results/ranking/weight_tuning.csv")
    args = p.parse_args()

    queries = json.load(open(args.queries))
    ranker = GiftRanker(args.index)

    if args.cmd == "pool":
        export_pool(ranker, queries, args.k, args.out)
        return

    lab = pd.read_csv(args.labels)
    lab = lab[pd.to_numeric(lab["grade"], errors="coerce").notna()]
    labels = {qid: dict(zip(g["parent_asin"].astype(str), g["grade"].astype(int))) for qid, g in lab.groupby("qid")}
    df = tune(ranker, queries, labels, args.k, args.step, [float(x) for x in args.lambdas.split(",")])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.head(10).to_string(index=False))
    best = df.iloc[0]
    print("\nBest weights:", {s: float(best[s]) for s in SIGNALS}, "mmr_lambda:", float(best["mmr_lambda"]))
    print("Put these into DEFAULT_WEIGHTS in CODE/ranking/ranker.py if they beat the current ones.")


if __name__ == "__main__":
    main()
