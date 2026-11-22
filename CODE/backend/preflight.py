"""Pre-demo check: are all the files there, and do they describe the same products?

Reads the same GC_* settings as the server (see config.py) and checks:

  * the product corpus is readable                        (GC_CORPUS)
  * the image lookup covers the corpus                    (GC_IMAGE_LOOKUP)
  * the FAISS index was built from this corpus            (GC_SEMANTIC_INDEX, row_to_asin.txt)
  * the BM25 index was built from this corpus             (GC_RANK_INDEX, catalog.parquet)
  * the knowledge-graph export exists                     (GC_KG_EXPORT, optional)
  * LLM credentials are configured                        (.env FOUNDRY_*, unless GC_LLM=offline)

An index built from a different corpus is the classic demo failure: retrieval
returns ASINs the server has no product record for, and cards silently vanish.

    python -m CODE.backend.preflight          # exit code 1 if anything blocks the demo
"""

import json
import os
import sys

from .config import Settings

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def corpus_asins(path):
    asins = set()
    with open(path, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                asin = json.loads(line).get("parent_asin")
                if asin:
                    asins.add(asin)
    return asins


def coverage(index_asins, corpus):
    if not index_asins:
        return 0.0, 0.0
    inside = len(index_asins & corpus)
    return inside / len(index_asins), inside / len(corpus)


def main():
    s = Settings.from_env()
    results = []

    def report(level, name, detail):
        results.append(level)
        print(f"[{level}] {name:<18} {detail}")

    print("Gift Concierge preflight\n")

    if not s.corpus.exists():
        report(FAIL, "corpus", f"not found: {s.corpus}")
        print("\nCannot continue without the corpus.")
        return 1
    corpus = corpus_asins(s.corpus)
    report(OK, "corpus", f"{len(corpus):,} products  ({s.corpus})")

    # images
    if not s.image_lookup:
        report(WARN, "images", "GC_IMAGE_LOOKUP not set: cards will show placeholders")
    elif not s.image_lookup.exists():
        report(WARN, "images", f"not found: {s.image_lookup}")
    else:
        with open(s.image_lookup, encoding="utf-8") as infile:
            images = json.load(infile)
        covered = len(corpus & set(images))
        share = covered / len(corpus)
        level = OK if share >= 0.5 else WARN
        report(level, "images", f"{covered:,} of {len(corpus):,} products have an image ({share:.0%})")

    # semantic index
    row_map = s.semantic_index / "row_to_asin.txt"
    if not (s.semantic_index / "index.faiss").exists() or not row_map.exists():
        level = FAIL if s.retriever == "semantic" else WARN
        report(level, "semantic index", f"no index.faiss/row_to_asin.txt in {s.semantic_index} "
                                        "(server would use the TF-IDF stand-in)")
    else:
        rows = set(row_map.read_text(encoding="utf-8").split())
        in_corpus, of_corpus = coverage(rows, corpus)
        level = OK if in_corpus > 0.99 and of_corpus > 0.95 else FAIL
        report(level, "semantic index", f"{len(rows):,} vectors; {in_corpus:.1%} in corpus, "
                                        f"covers {of_corpus:.1%} of it")

    # BM25 index
    catalog = s.rank_index / "catalog.parquet"
    if not catalog.exists():
        report(WARN, "bm25 index", f"no catalog.parquet in {s.rank_index} (server would use the stub ranker)")
    else:
        try:
            import pandas as pd
            asins = set(pd.read_parquet(catalog, columns=["parent_asin"])["parent_asin"].astype(str))
            in_corpus, of_corpus = coverage(asins, corpus)
            level = OK if in_corpus > 0.99 and of_corpus > 0.95 else FAIL
            report(level, "bm25 index", f"{len(asins):,} products; {in_corpus:.1%} in corpus, "
                                        f"covers {of_corpus:.1%} of it")
        except Exception as exc:
            report(WARN, "bm25 index", f"could not read {catalog}: {exc}")

    # knowledge graph
    if s.kg == "off":
        report(WARN, "knowledge graph", "GC_KG=off")
    elif s.kg_export:
        found = s.kg_export.exists() and any(s.kg_export.iterdir())
        report(OK if found else FAIL, "knowledge graph",
               f"export {'found' if found else 'missing'}: {s.kg_export}")
    else:
        level = WARN if len(corpus) > 5000 else OK
        report(level, "knowledge graph", "no GC_KG_EXPORT: the graph is rebuilt at every start"
               + (" (slow at this size; build an export once)" if level == WARN else ""))

    # LLM
    if s.llm == "offline":
        report(WARN, "llm", "GC_LLM=offline: rule-based demo extractor, not the real LLM")
    else:
        missing = [k for k in ("FOUNDRY_ENDPOINT", "FOUNDRY_API_KEY", "FOUNDRY_DEPLOYMENT") if not os.environ.get(k)]
        report(FAIL if missing else OK, "llm",
               f"missing {', '.join(missing)} in .env" if missing else f"Foundry deployment {os.environ['FOUNDRY_DEPLOYMENT']}")

    blocked = FAIL in results
    print("\n" + ("Fix the FAIL lines before the demo." if blocked
                  else "Ready. Warnings fall back gracefully and show up in /health."))
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
