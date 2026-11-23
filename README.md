# ClickLessAI: Gift Concierge

A conversational, evidence-grounded gift recommender (CSE573, Group 8). Describe who you're shopping for in plain language; Gift Concierge asks a follow-up question if it needs one, searches Amazon products from three domains (Books, Handmade, Toys & Games), and returns a ranked shortlist where every reason is backed by product data, reviews or knowledge-graph facts. You can keep refining in the same conversation ("cheaper", "she's not into reading anymore").

```
browser (CODE/frontend)
  └─ POST /recommend ─► backend (CODE/backend)
        └─ GiftConcierge (CODE/llm)            LLM constraint extraction, dialogue state, clarification
             ├─ semantic retrieval             BGE embeddings + FAISS        (CODE/retrieval)
             ├─ BM25 + RRF + filters + ranking + MMR                         (CODE/retrieval, CODE/ranking)
             ├─ knowledge-graph evidence                                     (CODE/knowledge_graph)
             └─ grounded explanation, then cards with images and Amazon links
```

## Quick Start / Grader Demo (900-product sample)

Everything below uses only files committed to this repository. The demo runs on the 900-product development sample `DATA/samples/sample_corpus_v2.jsonl` (300 each from Books, Handmade and Toys & Games). Generated indexes go to `artifacts/demo900/`, which is gitignored.

**Requirements:** Python 3.9 to 3.12, about 3 GB free disk (CPU PyTorch), 4 GB RAM, internet on the first run (pip packages and the ~130 MB BGE model from Hugging Face). macOS and Linux run the commands as shown; on Windows use WSL or Git Bash (see Troubleshooting).

```bash
# 1. Clone
git clone https://github.com/kkauserr/GiftConciergeAI.git
cd GiftConciergeAI

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows (Git Bash): source .venv/Scripts/activate

# 3. Dependencies
#    Linux/Windows without an NVIDIA GPU: install CPU PyTorch first (avoids ~2-3 GB of CUDA wheels)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Optional: real LLM. Without this step the app runs in offline mode (see below).
cp .env.example .env                 # then fill in FOUNDRY_ENDPOINT, FOUNDRY_API_KEY, FOUNDRY_DEPLOYMENT

# 5. Build the indexes (BM25, BGE + FAISS, knowledge graph), about 2-5 minutes on a laptop CPU
bash CODE/backend/demo900.sh prepare

# 6. Verify: every line should say ok (the llm line says warn in offline mode)
bash CODE/backend/demo900.sh check

# 7. Run
bash CODE/backend/demo900.sh serve
```

Open **http://localhost:8000** and try:

> I need a graduation gift for my sister. She's 22, loves coffee and reading. Budget is $80. Nothing wearable.

then follow up with *"Keep it under $50"*, *"She's actually not into reading anymore"*, or one of the suggested replies. **Start over** (top right) begins a new conversation. Stop the server with Ctrl+C.

### LLM: Foundry or offline

| Mode | When | What changes |
|---|---|---|
| **Foundry** | `.env` has `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `FOUNDRY_DEPLOYMENT` (see `.env.example`) | Real LLM constraint extraction and grounded explanations |
| **Offline** | No `.env` credentials (automatic), or `GC_LLM=offline` | A rule-based extractor stands in for the LLM; explanations are templated and start with `[offline]` |

`demo900.sh` picks the mode automatically (`GC_LLM=auto`). **Only the LLM is substituted:** semantic retrieval, BM25 ranking, filters, MMR, the knowledge graph, the backend and the frontend are the same in both modes, so graders without our API key can still use the full application. The offline extractor understands simpler phrasing than the real LLM (for example, it ignores "more unique"). Force a mode with `GC_LLM=foundry bash CODE/backend/demo900.sh serve` or `GC_LLM=offline …`.

### What `prepare`, `check` and `serve` do

| Command | Does |
|---|---|
| `prepare` | Checks Python and packages, then builds into `artifacts/demo900/`: BM25 index (`bm25/`), BGE embeddings + FAISS index (`semantic/`), knowledge-graph export (`kg/`). Steps already built are skipped; `prepare --force` deletes and rebuilds everything. |
| `check` | Python version and all required packages; the corpus; that the BM25, FAISS and KG artifacts each cover exactly the 900 products; product-image coverage; which LLM mode will be used. Exit code 1 on any FAIL. |
| `serve` | Runs `check`, then starts the backend, which also serves the frontend, on http://localhost:8000. `PORT=8001` changes the port. `http://localhost:8000/health` lists the live components. |

Product images come from `DATA/samples/sample_image_lookup.json` (built from the raw Amazon metadata with `CODE/data/build_image_lookup.py`). If that file is missing, cards show a category placeholder; everything else works.

### Expected time and resources

| Step | Typical time | Notes |
|---|---|---|
| `pip install` (CPU PyTorch) | 3-10 min | ~2.5 GB in the virtual environment |
| `prepare`, first run | 2-5 min | mostly the one-time BGE model download; embedding 900 products takes well under a minute on CPU |
| `prepare`, later runs | seconds | already-built steps are skipped |
| `serve` startup | 10-30 s | loads the BGE model and the indexes |
| Each message | 1-3 s offline; plus LLM latency in Foundry mode | |

Peak memory is under 2 GB. No GPU is needed for the 900-product demo.

## The two demo configurations

| | 900-product GitHub demo | 60K Sol demo |
|---|---|---|
| Corpus | `DATA/samples/sample_corpus_v2.jsonl` (committed) | `dev_subset_v2.jsonl`, 60,000 products (20K per domain) on ASU Sol scratch |
| Indexes | built locally by `demo900.sh prepare` into `artifacts/demo900/` | prebuilt on Sol (semantic, BM25, KG); not committed because of size |
| Images | `DATA/samples/sample_image_lookup.json` | `dev60k_image_lookup.json` on Sol |
| Script | `CODE/backend/demo900.sh` | `CODE/backend/dev60k.sh` |
| Reproducible from GitHub | yes | no: needs the Sol files |
| LLM | Foundry if `.env` is set, otherwise offline | Foundry (required) |

Both run exactly the same frontend, backend and pipeline code; only the data paths differ. Details for the 60K setup: `CODE/backend/README.md`.

## Python and dependencies

- **Python 3.9 to 3.12.** Our Sol environment runs Python 3.9; all code avoids 3.10+ syntax.
- **`requirements.txt`** installs everything the application needs: frontend server, backend, LLM layer, BGE/sentence-transformers, PyTorch, FAISS, BM25/ranking, knowledge graph, Parquet support. No other install step is required.
- **`requirements-dev.txt`** adds test and evaluation tools (`pytest`, `tabulate`): `pip install -r requirements-dev.txt`, then `python -m pytest CODE/tests`.
- `CODE/ranking/requirements.txt` and `CODE/knowledge_graph/requirements.txt` are small subsets for working on one component alone.
- **PyTorch:** pip's default PyTorch on Linux and Windows includes CUDA (several GB). Without an NVIDIA GPU, install the CPU build first as shown in step 3. macOS (Intel and Apple Silicon) needs nothing special.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check` says `missing …: pip install -r requirements.txt` | The virtual environment isn't active, or the install failed. Activate `.venv` and reinstall. |
| pip downloads gigabytes of `nvidia-*` packages | Install CPU PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `prepare` fails downloading `BAAI/bge-small-en-v1.5` | Needs internet access to huggingface.co on the first run. Behind a proxy, set `HTTPS_PROXY`. The model is cached afterwards (`~/.cache/huggingface`). |
| `check` shows FAIL for an index ("% in corpus") | The artifact was built from a different corpus or is incomplete: `bash CODE/backend/demo900.sh prepare --force` |
| macOS: `OMP: Error #15 … libomp already initialized` | PyTorch and FAISS each ship OpenMP. Run with `KMP_DUPLICATE_LIB_OK=TRUE bash CODE/backend/demo900.sh serve` |
| Port 8000 already in use | `PORT=8001 bash CODE/backend/demo900.sh serve`, then open http://localhost:8001 |
| The page works but answers look canned / start with `[offline]` | Offline mode: no Foundry credentials in `.env`. Expected for graders without a key. |
| Foundry errors in the server log | Check the three `FOUNDRY_*` values in `.env`; the deployment must support JSON-schema output. Or run `GC_LLM=offline`. |
| Product cards show placeholder icons | `DATA/samples/sample_image_lookup.json` is missing; images are optional. |
| Opening `CODE/frontend/index.html` directly shows a demo with only a few products | That's the frontend's built-in mock, used when no backend is running. Use http://localhost:8000 instead. |
| Windows | Run the commands in WSL (recommended) or Git Bash. `demo900.sh` is a bash script; `python` must be the virtual environment's Python. |

## Repository layout

| Path | Contents | Owner |
|---|---|---|
| `CODE/frontend/` | Conversational UI (HTML/CSS/JS) | Jameela |
| `CODE/backend/` | HTTP server, response assembly, `demo900.sh`, `dev60k.sh`, preflight | Jameela |
| `CODE/data/` | Amazon Reviews 2023 preprocessing, sample and image lookup builders | Jameela |
| `CODE/llm/`, `CODE/retrieval/` (semantic) | LLM layer, dialogue state, BGE + FAISS retrieval | Shashank G |
| `CODE/retrieval/bm25_baseline.py`, `CODE/ranking/` | BM25 baseline, RRF fusion, filters, ranking, MMR, ablation | Disha |
| `CODE/knowledge_graph/` | Ontology, graph build, evidence API | Jenna |
| `CODE/tests/` | LLM layer and backend tests | |
| `DATA/samples/` | 900-product sample corpus (and image lookup) | |
| `EVALUATIONS/` | Evaluation results | |
