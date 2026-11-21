"""Build the Gift Concierge KG from a corpus JSONL file and report on it.

Run from the CODE/ directory:
    python -m knowledge_graph.build_kg                        # sample, all inference
    python -m knowledge_graph.build_kg --no-inference         # foundation only
    python -m knowledge_graph.build_kg --no-review-inference  # listing inference only
    python -m knowledge_graph.build_kg --coverage --suspicious 50 --examples 2
    python -m knowledge_graph.build_kg --out kg_output        # write nodes/edges jsonl
    python -m knowledge_graph.build_kg --labeling-sheet new_sheet.csv   # refuses to overwrite
    python -m knowledge_graph.build_kg --review-sheet new_review_sheet.csv

The input defaults to DATA/samples/sample_corpus_v2.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import KnowledgeGraph, iter_jsonl
from .ontology import NodeType
from .reports import coverage, suspicious, write_labeling_sheet, write_review_sheet

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "DATA" / "samples" / "sample_corpus_v2.jsonl"


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _write_sheet(writer, kg, path, force, what) -> bool:
    try:
        n = writer(kg, path, overwrite=force)
    except FileExistsError as exc:
        print(f"\nNot written: {exc}")
        return False
    print(f"\nWrote {what} with {n} rows to {path}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    ap.add_argument("--limit", type=int, help="read only the first N records")
    ap.add_argument("--no-inference", action="store_true",
                    help="skip all interest/aspect/review inference (foundation graph only)")
    ap.add_argument("--no-review-inference", action="store_true",
                    help="skip SUPPORTS and review-derived HAS_ASPECT")
    ap.add_argument("--coverage", action="store_true", help="print per-domain coverage")
    ap.add_argument("--suspicious", type=int, metavar="N", default=0,
                    help="print up to N inferred edges flagged for manual review")
    ap.add_argument("--examples", type=int, default=0,
                    help="print N example product summaries per domain")
    ap.add_argument("--out", help="directory for nodes.jsonl / edges.jsonl")
    ap.add_argument("--labeling-sheet", metavar="CSV",
                    help="write the balanced 60-product interest/aspect sheet to a NEW CSV")
    ap.add_argument("--review-sheet", metavar="CSV",
                    help="write a stratified review-evidence sheet to a NEW CSV")
    ap.add_argument("--force", action="store_true",
                    help="allow the sheet options to overwrite an existing file")
    args = ap.parse_args(argv)

    kg = KnowledgeGraph.build(iter_jsonl(args.input, args.limit),
                              inference=not args.no_inference,
                              review_inference=not args.no_review_inference)
    stats = kg.stats()
    stats["vocabularies"] = kg.vocab_versions
    _dump(stats)

    problems = kg.validate()
    print(f"\nValidation: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for p in problems[:20]:
        print("  -", p)

    if args.coverage:
        print("\nCoverage by domain:")
        _dump(coverage(kg))

    if args.suspicious:
        flagged = suspicious(kg)
        print(f"\nSuspicious inferred edges: {len(flagged)} (showing {min(len(flagged), args.suspicious)})")
        for item in flagged[:args.suspicious]:
            print(" ", json.dumps(item, ensure_ascii=False))

    if args.examples:
        seen = {}
        for pid in kg.nodes_of_type(NodeType.PRODUCT):  # file order
            d = kg.g.nodes[pid]["domain"]
            if seen.get(d, 0) < args.examples:
                seen[d] = seen.get(d, 0) + 1
                _dump(kg.product_summary(kg.g.nodes[pid]["parent_asin"]))

    if args.out:
        paths = kg.export_jsonl(args.out)
        print(f"\nWrote {paths['nodes']} and {paths['edges']}")
    ok = True
    if args.labeling_sheet:
        ok &= _write_sheet(write_labeling_sheet, kg, args.labeling_sheet, args.force, "labeling sheet")
    if args.review_sheet:
        ok &= _write_sheet(write_review_sheet, kg, args.review_sheet, args.force, "review evidence sheet")
    if problems:
        return 1
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
