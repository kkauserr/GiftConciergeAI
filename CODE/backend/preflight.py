"""Pre-demo check: are all the files there, and do they describe the same products?

Reads the same GC_* settings as the server (see config.py) and checks:

  * the product corpus is readable                        (GC_CORPUS)
  * the image lookup covers the corpus                    (GC_IMAGE_LOOKUP)
  * the FAISS index was built from this corpus            (GC_SEMANTIC_INDEX, row_to_asin.txt)
  * the BM25 index was built from this corpus             (GC_RANK_INDEX, catalog.parquet)
  * the knowledge-graph export was built from this corpus (GC_KG_EXPORT, optional)
  * LLM credentials are configured                        (.env FOUNDRY_*; auto/offline fall back)

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


# import name -> pip package (all in the root requirements.txt)
REQUIRED_MODULES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "pyarrow": "pyarrow",
    "sklearn": "scikit-learn", "bm25s": "bm25s", "Stemmer": "PyStemmer",
    "torch": "torch", "sentence_transformers": "sentence-transformers", "faiss": "faiss-cpu",
    "openai": "openai", "pydantic": "pydantic", "dotenv": "python-dotenv",
    "networkx": "networkx", "starlette": "starlette", "uvicorn": "uvicorn",
}


def check_dependencies(report):
    import importlib.util

    version = sys.version_info
    report(OK if version >= (3, 9) else FAIL, "python",
           f"{sys.version.split()[0]} ({sys.executable})" + ("" if version >= (3, 9) else ": needs 3.9+"))
    missing = [pkg for mod, pkg in REQUIRED_MODULES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        report(FAIL, "packages", f"missing {', '.join(missing)}: pip install -r requirements.txt")
    else:
        report(OK, "packages", f"all {len(REQUIRED_MODULES)} required packages installed")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Gift Concierge preflight")
    parser.add_argument("--deps-only", action="store_true", help="only check Python and packages")
    args = parser.parse_args(argv)

    s = Settings.from_env()
    results = []

    def report(level, name, detail):
        results.append(level)
        print(f"[{level}] {name:<18} {detail}")

    print("Gift Concierge preflight\n")
    check_dependencies(report)
    if args.deps_only:
        if FAIL in results:
            print("\nInstall the missing packages first.")
            return 1
        return 0

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
        nodes = s.kg_export / "nodes.jsonl"
        if not nodes.exists():
            report(FAIL, "knowledge graph", f"export missing: no nodes.jsonl in {s.kg_export}")
        else:
            products = set()
            with open(nodes, encoding="utf-8") as infile:
                for line in infile:
                    node = json.loads(line)
                    if node.get("type") == "Product" and node.get("parent_asin"):
                        products.add(node["parent_asin"])
            in_corpus, of_corpus = coverage(products, corpus)
            level = OK if in_corpus > 0.99 and of_corpus > 0.95 else FAIL
            report(level, "knowledge graph", f"{len(products):,} products; {in_corpus:.1%} in corpus, "
                                             f"covers {of_corpus:.1%} of it")
    else:
        level = WARN if len(corpus) > 5000 else OK
        report(level, "knowledge graph", "no GC_KG_EXPORT: the graph is rebuilt at every start"
               + (" (slow at this size; build an export once)" if level == WARN else ""))

    # LLM
    missing = [k for k in ("FOUNDRY_ENDPOINT", "FOUNDRY_API_KEY", "FOUNDRY_DEPLOYMENT") if not os.environ.get(k)]
    if s.llm == "offline":
        report(WARN, "llm", "GC_LLM=offline: rule-based demo extractor, not the real LLM")
    elif not missing:
        report(OK, "llm", f"Foundry deployment {os.environ['FOUNDRY_DEPLOYMENT']}"
               + (" (auto)" if s.llm == "auto" else ""))
    elif s.llm == "auto":
        report(WARN, "llm", "no FOUNDRY_* in .env: offline demo extractor will be used "
                            "(retrieval, ranking and KG are unchanged)")
    else:
        report(FAIL, "llm", f"missing {', '.join(missing)} in .env")

    blocked = FAIL in results
    print("\n" + ("Fix the FAIL lines before the demo." if blocked
                  else "Ready. Warnings fall back gracefully and show up in /health."))
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
