"""Semantic retrieval: query text -> top-N products by cosine similarity.

Output (the handoff to the BM25/RRF/filtering/ranking stage):

    [{"parent_asin": "B07...", "score": 0.7412, "rank": 1}, ...]

`rank` starts at 1; `score` is the cosine similarity (inner product of
normalized BGE vectors), higher is better, sorted descending.

CLI:
    python CODE/retrieval/semantic_search.py "dinosaur puzzle for a 6 year old" --top-n 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import (  # noqa: E402
    EMBED_DIM,
    INDEX_FILE,
    MANIFEST_FILE,
    QUERY_PREFIX,
    ROW_MAP_FILE,
    SAMPLE_INDEX_DIR,
)


class SemanticRetriever:
    def __init__(self, index_dir=SAMPLE_INDEX_DIR, encoder=None, device=None):
        import faiss

        index_dir = Path(index_dir)
        if not (index_dir / INDEX_FILE).exists():
            raise FileNotFoundError(
                f"No {INDEX_FILE} in {index_dir}. Run embed_corpus.py and "
                "build_faiss_index.py first."
            )
        self.manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / INDEX_FILE))
        self.row_to_asin = (index_dir / ROW_MAP_FILE).read_text(encoding="utf-8").split()
        if self.index.ntotal != len(self.row_to_asin):
            raise ValueError(
                f"Index has {self.index.ntotal} rows but row map has {len(self.row_to_asin)}")
        if self.index.d != EMBED_DIM:
            raise ValueError(f"Index dim {self.index.d} != {EMBED_DIM}")
        self.query_prefix = self.manifest.get("query_prefix", QUERY_PREFIX)

        if encoder is None:
            from retrieval.embed_corpus import load_encoder
            encoder = load_encoder(self.manifest["model"], device=device)
        self._encode = encoder

    @property
    def size(self):
        return self.index.ntotal

    def embed_queries(self, queries):
        texts = [self.query_prefix + q.strip() for q in queries]
        return np.ascontiguousarray(self._encode(texts, 64), dtype=np.float32)

    def search_batch(self, queries, top_n=100):
        for q in queries:
            if not q or not q.strip():
                raise ValueError("query_text must be non-empty")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        k = min(top_n, self.size)
        scores, rows = self.index.search(self.embed_queries(queries), k)
        results = []
        for score_row, id_row in zip(scores, rows):
            hits = []
            for score, row in zip(score_row, id_row):
                if row < 0:  # FAISS pads with -1 when fewer than k results
                    continue
                hits.append({
                    "parent_asin": self.row_to_asin[row],
                    "score": round(float(score), 6),
                    "rank": len(hits) + 1,
                })
            results.append(hits)
        return results

    def search(self, query_text, top_n=100):
        """Return [{parent_asin, score, rank}] for the top_n most similar products."""
        return self.search_batch([query_text], top_n=top_n)[0]


def main():
    parser = argparse.ArgumentParser(description="Semantic search over embed_text")
    parser.add_argument("query")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--index-dir", default=str(SAMPLE_INDEX_DIR))
    args = parser.parse_args()
    retriever = SemanticRetriever(args.index_dir)
    print(json.dumps(retriever.search(args.query, top_n=args.top_n), indent=2))


if __name__ == "__main__":
    main()
