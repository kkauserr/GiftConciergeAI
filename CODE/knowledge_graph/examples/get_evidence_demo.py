"""Minimal end-to-end use of the KG integration API (no LLM required).

From CODE/:
    python -m knowledge_graph.examples.get_evidence_demo
    python -m knowledge_graph.examples.get_evidence_demo --asins B0164RRP14 1683239822 --interests dogs
    python -m knowledge_graph.examples.get_evidence_demo --json          # full JSON response
    python -m knowledge_graph.examples.get_evidence_demo --export-dir kg_output   # load a prebuilt export

The default ASINs show each kind of result: review-supported qualities,
a review concern, a product with no interest match, and an unknown ASIN.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from knowledge_graph.api import evidence_to_text, load_kg

DEFAULT_ASINS = ["B0164RRP14", "B09XDX9Q8T", "B07BLL7WQD", "1683239822", "NOT_A_REAL_ASIN"]
DEFAULT_INTERESTS = ["fishing_hunting", "puzzles", "ponies"]   # "ponies" is not a label


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asins", nargs="*", default=DEFAULT_ASINS)
    ap.add_argument("--interests", nargs="*", default=DEFAULT_INTERESTS)
    ap.add_argument("--max-reviews", type=int, default=2)
    ap.add_argument("--export-dir", help="load a graph written by build_kg --out DIR")
    ap.add_argument("--json", action="store_true", help="print the full JSON response")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    kg = load_kg(export_dir=args.export_dir) if args.export_dir else load_kg()   # once per process
    t1 = time.perf_counter()
    resp = kg.get_evidence(args.asins, query_interests=args.interests, max_reviews=args.max_reviews)
    t2 = time.perf_counter()
    kg.get_evidence(args.asins, query_interests=args.interests)    # repeated call: no rebuild
    t3 = time.perf_counter()

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return 0
    info = kg.info()
    print(f"Loaded {info['source']} in {t1 - t0:.1f}s  ({info['nodes']['Product']} products, "
          f"contract {info['contract_version']}, {info['vocabulary']})")
    print(f"get_evidence: {1000 * (t2 - t1):.1f} ms, repeated call {1000 * (t3 - t2):.1f} ms\n")
    print("Accepted query interests:", resp["query"]["query_interests"])
    for w in resp["warnings"]:
        print(f"warning {w['code']}: {w['value']}")
    print()
    for r in resp["results"]:
        print(evidence_to_text(r))
        m = r.get("query_interest_match")
        if m:
            print(f"  query interests matched: {m['matched'] or 'none'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
