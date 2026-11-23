"""Backend settings, read from environment variables (or a repo-root .env).

Every path defaults to the 900-product development sample so the server runs
locally with no setup beyond the pip installs. On Sol, point the paths at the
full-corpus files.

    GC_LLM               "foundry" (default; needs FOUNDRY_* in .env) or "offline"
                         (Shashank's rule-based demo extractor; never for evaluation)
    GC_CORPUS            product corpus JSONL               [DATA/samples/sample_corpus_v2.jsonl]
    GC_SEMANTIC_INDEX    FAISS index dir from embed_corpus  [DATA/embeddings/sample_v2]
    GC_RETRIEVER         "auto" (default), "semantic" or "tfidf"
                         auto = semantic when FAISS + the index are available,
                         otherwise a TF-IDF stand-in (reported by /health)
    GC_RANK_INDEX        BM25 index dir from CODE/retrieval/bm25_baseline.py   [indexes/sample]
                         used with Disha's GiftRanker when it exists, else the stub ranker
    GC_KG                "auto" (default), "on" or "off": Jenna's knowledge-graph evidence
    GC_KG_EXPORT         optional prebuilt KG export dir (recommended for the full corpus)
    GC_IMAGE_LOOKUP      JSON {parent_asin: image_url} from CODE/data/build_image_lookup.py
                         [DATA/images/image_lookup.json if present]
    GC_TOP_K             recommendations per turn           [5]
    GC_SESSION_TTL_MIN   idle minutes before a conversation is dropped   [120]
    GC_MAX_SESSIONS      conversations kept in memory       [200]
    GC_CORS_ORIGINS      comma-separated origins, or "*"    [*]
    GC_DEBUG             "1" adds a `debug` block (state changes, query, ranking) to responses
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "CODE"
FRONTEND_DIR = CODE_DIR / "frontend"


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def _path(name, default):
    value = os.environ.get(name)
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return default


def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    llm: str = "foundry"
    corpus: Path = REPO_ROOT / "DATA" / "samples" / "sample_corpus_v2.jsonl"
    semantic_index: Path = REPO_ROOT / "DATA" / "embeddings" / "sample_v2"
    retriever: str = "auto"
    rank_index: Path = REPO_ROOT / "indexes" / "sample"
    kg: str = "auto"
    kg_export: Optional[Path] = None
    image_lookup: Optional[Path] = None
    top_k: int = 5
    session_ttl_minutes: int = 120
    max_sessions: int = 200
    cors_origins: list = field(default_factory=lambda: ["*"])
    debug: bool = False

    @classmethod
    def from_env(cls):
        _load_dotenv()
        default_images = REPO_ROOT / "DATA" / "images" / "image_lookup.json"
        origins = os.environ.get("GC_CORS_ORIGINS", "*")
        return cls(
            llm=os.environ.get("GC_LLM", "foundry").strip().lower(),
            corpus=_path("GC_CORPUS", cls.corpus),
            semantic_index=_path("GC_SEMANTIC_INDEX", cls.semantic_index),
            retriever=os.environ.get("GC_RETRIEVER", "auto").strip().lower(),
            rank_index=_path("GC_RANK_INDEX", cls.rank_index),
            kg=os.environ.get("GC_KG", "auto").strip().lower(),
            kg_export=_path("GC_KG_EXPORT", None),
            image_lookup=_path("GC_IMAGE_LOOKUP", default_images if default_images.exists() else None),
            top_k=int(os.environ.get("GC_TOP_K", "5")),
            session_ttl_minutes=int(os.environ.get("GC_SESSION_TTL_MIN", "120")),
            max_sessions=int(os.environ.get("GC_MAX_SESSIONS", "200")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            debug=_flag("GC_DEBUG"),
        )
