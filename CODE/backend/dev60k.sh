#!/usr/bin/env bash
# Gift Concierge class demo on the 60K dev subset (runs on ASU Sol).
#
#   bash CODE/backend/dev60k.sh prepare   # one-time: BM25 index + knowledge-graph export
#   bash CODE/backend/dev60k.sh check     # preflight: files exist and match the corpus
#   bash CODE/backend/dev60k.sh serve     # check, then start the server on port 8000
#
# Run from the repo root on Sol. Every path below can be overridden by exporting the
# variable first, e.g.  GC_SEMANTIC_INDEX=/scratch/sgadipal/... bash CODE/backend/dev60k.sh serve
set -euo pipefail

cd "$(dirname "$0")/../.."     # repo root

GC_ROOT="${GC_ROOT:-/scratch/jkauser/gift-concierge}"

# The 60K corpus and its image lookup (from make_dev_subset.py / build_image_lookup.py)
export GC_CORPUS="${GC_CORPUS:-$GC_ROOT/data/dev_subset_v2.jsonl}"
export GC_IMAGE_LOOKUP="${GC_IMAGE_LOOKUP:-$GC_ROOT/data/dev60k_image_lookup.json}"
# Built from the SAME corpus: semantic index (Shashank), BM25 index (Disha), KG export (Jenna)
export GC_SEMANTIC_INDEX="${GC_SEMANTIC_INDEX:-$GC_ROOT/embeddings/dev60k_v2}"
export GC_RANK_INDEX="${GC_RANK_INDEX:-indexes/dev60k}"   # Disha's existing 60K index (repo-relative)
export GC_KG_EXPORT="${GC_KG_EXPORT:-$GC_ROOT/kg_dev60k}"
# Demo: require the real components instead of silently falling back
export GC_RETRIEVER="${GC_RETRIEVER:-semantic}"
export GC_KG="${GC_KG:-on}"
export GC_LLM="${GC_LLM:-foundry}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"        # reachable through an SSH tunnel to the compute node

prepare() {
    echo "== BM25 index -> $GC_RANK_INDEX"
    if [ -f "$GC_RANK_INDEX/catalog.parquet" ]; then
        echo "   already built"
    else
        python -m CODE.retrieval.bm25_baseline build --corpus "$GC_CORPUS" --out "$GC_RANK_INDEX" --skip-tfidf
    fi

    echo "== Knowledge-graph export -> $GC_KG_EXPORT"
    if [ -d "$GC_KG_EXPORT" ] && [ -n "$(ls -A "$GC_KG_EXPORT" 2>/dev/null)" ]; then
        echo "   already built"
    else
        (cd CODE && python -m knowledge_graph.build_kg "$GC_CORPUS" --out "$GC_KG_EXPORT")
    fi

    echo "== Semantic index -> $GC_SEMANTIC_INDEX"
    if [ -f "$GC_SEMANTIC_INDEX/index.faiss" ]; then
        echo "   already built"
    else
        echo "   not built yet. Shashank's commands (GPU node):"
        echo "     python CODE/retrieval/embed_corpus.py --input $GC_CORPUS --out-dir $GC_SEMANTIC_INDEX --device cuda --fp16"
        echo "     python CODE/retrieval/build_faiss_index.py --index-dir $GC_SEMANTIC_INDEX"
    fi
}

check() {
    python -m CODE.backend.preflight
}

serve() {
    check
    node="$(hostname)"
    cat <<EOF

Starting Gift Concierge on $node:$PORT
From your laptop, open a tunnel through the Sol login node, then browse to http://localhost:$PORT

    ssh -L $PORT:$node:$PORT ${USER}@login.sol.rc.asu.edu

EOF
    exec python -m uvicorn CODE.backend.app:app --host "$HOST" --port "$PORT"
}

case "${1:-}" in
    prepare) prepare ;;
    check)   check ;;
    serve)   serve ;;
    *) echo "usage: bash CODE/backend/dev60k.sh {prepare|check|serve}"; exit 2 ;;
esac
