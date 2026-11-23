# Gift Concierge backend

A thin HTTP layer between the frontend (`CODE/frontend`) and the recommendation pipeline. It holds one conversation per browser session and turns each `GiftConcierge.handle()` result into product cards with images and Amazon links. It makes no recommendation decisions itself.

```
browser ──POST /recommend──► app.py ──► GiftConcierge.handle()      (CODE/llm, Shashank)
                                │          ├─ SemanticRetriever       (CODE/retrieval, Shashank)
                                │          ├─ GiftRanker via adapter   (CODE/ranking, Disha)
                                │          └─ KG evidence via adapter  (CODE/knowledge_graph, Jenna)
                                └──► assemble.py: product details + image_url + amazon_url
```

The components are wired the same way as `CODE/llm/demo_cli.py`.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux/Windows without an NVIDIA GPU
pip install -r requirements.txt                                      # from the repo root: the whole application
```

The root `requirements.txt` covers everything the backend serves (LLM layer, BGE + FAISS, BM25 ranking, knowledge graph, Starlette/Uvicorn). Python 3.9 to 3.12. `CODE/backend/requirements.txt` just points to it.

## Grader demo: 900-product sample (`demo900.sh`)

Fully reproducible from committed files; see the **Quick Start** in the root `README.md` for the full walkthrough from clone to browser.

```bash
bash CODE/backend/demo900.sh prepare    # BM25, BGE + FAISS and KG into artifacts/demo900/ (gitignored)
bash CODE/backend/demo900.sh check      # packages + every artifact covers the 900 products
bash CODE/backend/demo900.sh serve      # http://localhost:8000
```

| What | Path |
|---|---|
| Corpus | `DATA/samples/sample_corpus_v2.jsonl` |
| Image lookup | `DATA/samples/sample_image_lookup.json` (optional; placeholders without it) |
| BM25 index | `artifacts/demo900/bm25` |
| Semantic index | `artifacts/demo900/semantic` |
| KG export | `artifacts/demo900/kg` |

`demo900.sh` uses `GC_LLM=auto`: Foundry when `.env` has `FOUNDRY_*` credentials, otherwise Shashank's offline extractor. Only the LLM changes; retrieval, ranking, KG, backend and frontend are identical. `prepare --force` rebuilds every artifact from scratch.

## Run manually

```bash
uvicorn CODE.backend.app:app --port 8000                  # settings from GC_* variables (config.py)
GC_LLM=offline uvicorn CODE.backend.app:app --port 8000   # without Foundry credentials
```

Open **http://localhost:8000**. The page detects the backend on its own; a page opened as a file uses the mock instead.

Check **http://localhost:8000/health** to see which components are live. When run manually (default `GC_RETRIEVER=auto`, `GC_KG=auto`), anything missing falls back and is reported there. `demo900.sh` and `dev60k.sh` instead require the semantic index and the KG.

| Component | When available | Fallback |
|---|---|---|
| LLM | `FOUNDRY_*` in `.env` | `GC_LLM=auto` or `offline`: rule-based extractor |
| Semantic retrieval | FAISS + sentence-transformers + index in `GC_SEMANTIC_INDEX` | TF-IDF stand-in (dev only) |
| Ranking | BM25 index in `GC_RANK_INDEX` and `bm25s` installed | the LLM layer's stub ranker |
| Knowledge graph | `networkx` installed | explanations from product data only |
| Images | `GC_IMAGE_LOOKUP` file | placeholder icon on the card |

For evaluation or the demo, `/health` should show `foundry`, `semantic`, `GiftRanker` and `kg: on`.

## Class demo on Sol (60K corpus)

Not reproducible from GitHub: the corpus and its indexes live on Sol because of their size. The demo runs on the 60K dev subset (20K each from Books, Handmade and Toys & Games), which lives on Sol, not in Git. `dev60k.sh` holds the paths:

| What | Default path | Built by |
|---|---|---|
| Corpus | `/scratch/jkauser/gift-concierge/data/dev_subset_v2.jsonl` | `make_dev_subset.py` |
| Image lookup | `/scratch/jkauser/gift-concierge/data/dev60k_image_lookup.json` | `build_image_lookup.py` |
| Semantic index | `/scratch/jkauser/gift-concierge/indexes/semantic-dev60k` | `embed_corpus.py` + `build_faiss_index.py` (Shashank) |
| BM25 index | `indexes/dev60k` (in the repo checkout) | `bm25_baseline build` (Disha) |
| KG export | `/scratch/jkauser/gift-concierge/kg_dev60k` | `knowledge_graph.build_kg --out` (Jenna) |

Override any of them by exporting the variable first (for example, `GC_SEMANTIC_INDEX=/scratch/<user>/…`). All three indexes must be built from the same `dev_subset_v2.jsonl`.

On Sol, from the repo root (with the merged branches and the pip installs above):

```bash
bash CODE/backend/dev60k.sh prepare   # once: builds the BM25 index and KG export if missing,
                                      # prints Shashank's commands if the semantic index is missing
bash CODE/backend/dev60k.sh check     # preflight; every line should say ok
bash CODE/backend/dev60k.sh serve     # starts on port 8000 and prints the SSH tunnel command
```

Run `serve` inside an interactive job (for example `interactive -t 0-4 --mem=32G`). The tunnel goes from your laptop through the login node to that compute node:

```bash
ssh -L 8000:<compute-node>:8000 jkauser@login.sol.rc.asu.edu    # then open http://localhost:8000
```

`dev60k.sh` requires the real components (`GC_RETRIEVER=semantic`, `GC_KG=on`, `GC_LLM=foundry`), so a missing piece stops the server at startup instead of quietly using a stand-in. The preflight also catches an index built from a different corpus: retrieval would return products the server has no record for, and cards would silently disappear.

Before the demo, check two things:
- `.env` with `FOUNDRY_*` is in the repo root on Sol.
- The compute node can reach the Foundry endpoint. If compute nodes have no outbound internet, run `serve` on a node that does.

## Product images

Cards show the image whose URL is in a lookup file `{parent_asin: image_url}` written by `CODE/data/build_image_lookup.py` from the raw Amazon metadata (on Sol). The frontend only renders `image_url`; the lookup is read by the backend.

```bash
# 60K demo (defaults) -> /scratch/jkauser/gift-concierge/data/dev60k_image_lookup.json
python CODE/data/build_image_lookup.py
# 900-product sample -> commit the small output so graders get images
python CODE/data/build_image_lookup.py --subset DATA/samples/sample_corpus_v2.jsonl \
    --out DATA/samples/sample_image_lookup.json
```

`GC_IMAGE_LOOKUP` points the server at a lookup file; the lookup must cover the corpus being served. Products without an image show the category placeholder. "View on Amazon" links always work, since they are built from `parent_asin`.

## Settings

All settings are environment variables, documented at the top of `config.py`: `GC_LLM`, `GC_CORPUS`, `GC_SEMANTIC_INDEX`, `GC_RETRIEVER`, `GC_RANK_INDEX`, `GC_KG`, `GC_KG_EXPORT`, `GC_IMAGE_LOOKUP`, `GC_TOP_K`, `GC_SESSION_TTL_MIN`, `GC_MAX_SESSIONS`, `GC_CORS_ORIGINS`, `GC_DEBUG`.

`GC_DEBUG=1` adds a `debug` block to each response with state changes, conflicts, the query text, the ranking and the cited evidence IDs. The frontend ignores it; it's useful for evaluation and for checking faithfulness.

## Endpoints

| Endpoint | Body | Returns |
|---|---|---|
| `POST /recommend` | `{"session_id", "message"}` | `type: "question"` or `type: "recommendations"` (format in `CODE/frontend/README.md`) |
| `POST /reset` | `{"session_id"}` | `{"ok": true}`, forgets the conversation |
| `GET /health` | | component status |
| `GET /` | | the frontend |

Errors are JSON `{"error": "..."}`: 400 for a bad request, 503 when the engine couldn't start, 500 when the pipeline fails on a message (the conversation is kept).

## Tests

`CODE/tests/test_backend.py` runs a real `GiftConcierge` conversation through the engine with the offline LLM on the 900-product sample, so it needs no credentials or indexes. It checks the component report, the card fields (price, image, Amazon link), budget refinement, that evidence IDs stay out of responses, image lookup formats, and the session store.

```bash
python -m pytest CODE/tests/test_backend.py
```

## Notes and limits

- Conversations live in server memory: restarting the server, or running more than one worker, starts them over. Idle conversations are dropped after `GC_SESSION_TTL_MIN` minutes.
- `GiftConcierge` takes a dict of all products. That's fine for the sample and a 60K dev subset, but the full 27 GB corpus won't fit in memory this way; that needs a keyed product store in the LLM layer.
- Turns in one conversation run one at a time; different conversations run in parallel.
