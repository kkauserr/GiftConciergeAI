"""Shared settings for semantic retrieval.

Paths default to the development sample so everything runs locally. On Sol,
pass the full-corpus paths on the command line (see CODE/README.md).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
TEXT_FIELD = "embed_text"

# BGE v1.5 recommends this instruction for short retrieval queries.
# Documents (embed_text) are encoded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SAMPLE_CORPUS = REPO_ROOT / "DATA" / "samples" / "sample_corpus_v2.jsonl"
SAMPLE_INDEX_DIR = REPO_ROOT / "DATA" / "embeddings" / "sample_v2"

SOL_CORPUS = "/scratch/jkauser/gift-concierge/data/final_product_corpus_v2.jsonl"

# Files inside an index directory.
PROGRESS_FILE = "progress.json"
SHARD_EMB = "emb_{:05d}.npy"
SHARD_ASINS = "asins_{:05d}.txt"
INDEX_FILE = "index.faiss"
ROW_MAP_FILE = "row_to_asin.txt"
MANIFEST_FILE = "manifest.json"
