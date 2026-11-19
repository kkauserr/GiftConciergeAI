"""Merge embedding shards into a FAISS IndexFlatIP plus a row -> parent_asin map.

Reads the shards written by embed_corpus.py and writes, in the same directory:

    index.faiss       IndexFlatIP over normalized vectors (inner product = cosine)
    row_to_asin.txt   line i is the parent_asin of FAISS row i
    manifest.json     model, dim, row count, query prefix, build time

Local (sample):
    python CODE/retrieval/build_faiss_index.py

Sol:
    python CODE/retrieval/build_faiss_index.py \
        --index-dir /scratch/$USER/gift-concierge/embeddings/full_v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import (  # noqa: E402
    EMBED_DIM,
    INDEX_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    QUERY_PREFIX,
    ROW_MAP_FILE,
    SAMPLE_INDEX_DIR,
    SHARD_ASINS,
    SHARD_EMB,
)


def build_index(index_dir, allow_partial=False, log=print):
    import faiss

    index_dir = Path(index_dir)
    progress = json.loads((index_dir / PROGRESS_FILE).read_text(encoding="utf-8"))
    if not progress["finished"] and not allow_partial:
        raise RuntimeError(
            f"Embedding is not finished ({progress['rows_done']} rows so far). "
            "Rerun embed_corpus.py, or pass --allow-partial."
        )

    index = faiss.IndexFlatIP(EMBED_DIM)
    seen = set()
    duplicates = 0

    with open(index_dir / ROW_MAP_FILE, "w", encoding="utf-8", newline="\n") as row_map:
        for shard_id in range(progress["completed_shards"]):
            vectors = np.load(index_dir / SHARD_EMB.format(shard_id))
            asins = (index_dir / SHARD_ASINS.format(shard_id)).read_text(
                encoding="utf-8").split()
            if len(asins) != vectors.shape[0]:
                raise ValueError(
                    f"Shard {shard_id}: {vectors.shape[0]} vectors but {len(asins)} asins")
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            for asin in asins:
                if asin in seen:
                    duplicates += 1
                seen.add(asin)
                row_map.write(asin + "\n")
            log(f"added shard {shard_id:05d} ({index.ntotal} rows)")

    if index.ntotal != progress["rows_done"]:
        raise ValueError(f"Index has {index.ntotal} rows, expected {progress['rows_done']}")

    faiss.write_index(index, str(index_dir / INDEX_FILE))
    manifest = {
        "model": progress["model"],
        "text_field": progress["text_field"],
        "dim": EMBED_DIM,
        "num_rows": index.ntotal,
        "duplicate_asins": duplicates,
        "index_type": "IndexFlatIP",
        "normalized": True,
        "query_prefix": QUERY_PREFIX,
        "source": progress["input"],
        "complete": progress["finished"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote {INDEX_FILE} with {index.ntotal} rows "
        f"({duplicates} duplicate asins) to {index_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index-dir", default=str(SAMPLE_INDEX_DIR))
    parser.add_argument("--allow-partial", action="store_true",
                        help="build from completed shards even if embedding is unfinished")
    args = parser.parse_args()
    build_index(args.index_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
