#!/usr/bin/env bash
# Gift Concierge grader demo on the committed 900-product sample.
# Uses only files in this repository; everything it builds goes to artifacts/demo900/ (gitignored).
#
#   bash CODE/backend/demo900.sh prepare [--force]   # build BM25, BGE+FAISS and KG indexes (~2-5 min on CPU)
#   bash CODE/backend/demo900.sh check               # dependencies + every artifact matches the corpus
#   bash CODE/backend/demo900.sh serve               # start the app at http://localhost:8000
#
# LLM: uses Azure AI Foundry when FOUNDRY_* is set in .env (see .env.example); without it,
# falls back to the offline rule-based extractor. Only the LLM is substituted: retrieval,
# ranking, knowledge graph, backend and frontend are the same in both modes.
set -euo pipefail

cd "$(dirname "$0")/../.."     # repo root
ROOT="$(pwd)"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

ARTIFACTS="${ARTIFACTS:-artifacts/demo900}"

export GC_CORPUS="${GC_CORPUS:-DATA/samples/sample_corpus_v2.jsonl}"
export GC_IMAGE_LOOKUP="${GC_IMAGE_LOOKUP:-DATA/samples/sample_image_lookup.json}"   # committed, built on Sol
export GC_RANK_INDEX="${GC_RANK_INDEX:-$ARTIFACTS/bm25}"
export GC_SEMANTIC_INDEX="${GC_SEMANTIC_INDEX:-$ARTIFACTS/semantic}"
export GC_KG_EXPORT="${GC_KG_EXPORT:-$ARTIFACTS/kg}"
export GC_RETRIEVER="${GC_RETRIEVER:-semantic}"   # require the real semantic index
export GC_KG="${GC_KG:-on}"
export GC_LLM="${GC_LLM:-auto}"                   # Foundry if .env has credentials, else offline

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

prepare() {
    if [ "${1:-}" = "--force" ]; then
        echo "== Removing $ARTIFACTS"
        rm -rf "$ARTIFACTS"
    fi
    mkdir -p "$ARTIFACTS"
    "$PYTHON" -m CODE.backend.preflight --deps-only

    echo
    echo "== 1/3 BM25 index -> $GC_RANK_INDEX"
    if [ -f "$GC_RANK_INDEX/catalog.parquet" ]; then
        echo "   already built (use --force to rebuild)"
    else
        "$PYTHON" -m CODE.retrieval.bm25_baseline build --corpus "$GC_CORPUS" --out "$GC_RANK_INDEX" --skip-tfidf
    fi

    echo
    echo "== 2/3 BGE embeddings + FAISS index -> $GC_SEMANTIC_INDEX"
    if [ -f "$GC_SEMANTIC_INDEX/index.faiss" ]; then
        echo "   already built (use --force to rebuild)"
    else
        echo "   first run downloads BAAI/bge-small-en-v1.5 (~130 MB) from Hugging Face"
        "$PYTHON" CODE/retrieval/embed_corpus.py --input "$GC_CORPUS" --out-dir "$GC_SEMANTIC_INDEX"
        "$PYTHON" CODE/retrieval/build_faiss_index.py --index-dir "$GC_SEMANTIC_INDEX"
    fi

    echo
    echo "== 3/3 Knowledge-graph export -> $GC_KG_EXPORT"
    if [ -f "$GC_KG_EXPORT/nodes.jsonl" ]; then
        echo "   already built (use --force to rebuild)"
    else
        (cd CODE && "$PYTHON" -m knowledge_graph.build_kg "$ROOT/$GC_CORPUS" --out "$ROOT/$GC_KG_EXPORT")
    fi

    echo
    echo "== Product images"
    if [ -f "$GC_IMAGE_LOOKUP" ]; then
        echo "   using $GC_IMAGE_LOOKUP"
    else
        echo "   $GC_IMAGE_LOOKUP not found: cards will show placeholder images"
    fi

    echo
    echo "Done. Next: bash CODE/backend/demo900.sh check"
}

check() {
    "$PYTHON" -m CODE.backend.preflight
}

serve() {
    check || { echo "Fix the FAIL lines above (usually: bash CODE/backend/demo900.sh prepare)."; exit 1; }
    echo
    echo "Gift Concierge: open http://localhost:$PORT   (Ctrl+C to stop)"
    echo
    exec "$PYTHON" -m uvicorn CODE.backend.app:app --host "$HOST" --port "$PORT"
}

case "${1:-}" in
    prepare) prepare "${2:-}" ;;
    check)   check ;;
    serve)   serve ;;
    *) echo "usage: bash CODE/backend/demo900.sh {prepare [--force]|check|serve}"; exit 2 ;;
esac
