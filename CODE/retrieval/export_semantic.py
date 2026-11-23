"""Export semantic results for a query file, for the RRF fusion / ablation stage.

    python CODE/retrieval/export_semantic.py --queries CODE/ranking/dev_queries.json \
        --out semantic_dev.json
    python -m CODE.ranking.ablation --semantic-file semantic_dev.json

Input:  [{"qid", "constraints": {...}}, ...]
Output: {qid: [[parent_asin, score], ...]}, best first, top 200 per query.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.constraints import GiftConstraints, build_query_text  # noqa: E402
from retrieval.config import SAMPLE_INDEX_DIR  # noqa: E402


def query_text(constraints):
    """Same query builder as the live loop, from the shared constraints JSON."""
    return build_query_text(GiftConstraints(
        recipient=constraints.get("recipient"),
        occasion=constraints.get("occasion"),
        interests=list(constraints.get("interests") or []) + list(constraints.get("keywords") or []),
        age=constraints.get("recipient_age"),
    ))


def export(queries, retriever, top_n=200):
    hits = retriever.search_batch([query_text(q["constraints"]) for q in queries], top_n=top_n)
    return {q["qid"]: [[h["parent_asin"], h["score"]] for h in row]
            for q, row in zip(queries, hits)}


def main():
    parser = argparse.ArgumentParser(description="Export semantic top-N per query")
    parser.add_argument("--queries", default="CODE/ranking/dev_queries.json")
    parser.add_argument("--index-dir", default=str(SAMPLE_INDEX_DIR))
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from retrieval.semantic_search import SemanticRetriever
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    results = export(queries, SemanticRetriever(args.index_dir), args.top_n)
    Path(args.out).write_text(json.dumps(results), encoding="utf-8")
    print(f"{len(results)} queries -> {args.out}")


if __name__ == "__main__":
    main()
