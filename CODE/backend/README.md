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

## Requirements

The backend needs the LLM layer from the `shashank-g/semantic-llm` branch. Without it, the server still starts and serves the frontend, but `/health` and `/recommend` say the layer is missing, and the chat shows an error message.

```bash
pip install -r requirements.txt \
            -r CODE/backend/requirements.txt \
            -r CODE/ranking/requirements.txt \
            -r CODE/knowledge_graph/requirements.txt
```

## Run (from the repo root)

```bash
# one-time builds for the 900-product sample
python CODE/retrieval/embed_corpus.py          # semantic index -> DATA/embeddings/sample_v2
python CODE/retrieval/build_faiss_index.py
python -m CODE.retrieval.bm25_baseline build   # BM25 index    -> indexes/sample

# start (real LLM: fill in FOUNDRY_* in .env first)
uvicorn CODE.backend.app:app --port 8000
# no credentials yet: rule-based demo extractor (not for evaluation)
GC_LLM=offline uvicorn CODE.backend.app:app --port 8000
```

Open **http://localhost:8000**. The page detects the backend on its own; a page opened as a file uses the mock instead.

Check **http://localhost:8000/health** to see which components are live. Anything missing falls back and is reported there:

| Component | When available | Fallback |
|---|---|---|
| LLM | `FOUNDRY_*` in `.env` | none: set `GC_LLM=offline` explicitly |
| Semantic retrieval | FAISS + sentence-transformers + index in `GC_SEMANTIC_INDEX` | TF-IDF stand-in (dev only) |
| Ranking | BM25 index in `GC_RANK_INDEX` and `bm25s` installed | the LLM layer's stub ranker |
| Knowledge graph | `networkx` installed | explanations from product data only |
| Images | `GC_IMAGE_LOOKUP` file | placeholder icon on the card |

For evaluation or the demo, `/health` should show `foundry`, `semantic`, `GiftRanker` and `kg: on`.

## Class demo on Sol (60K corpus)

The demo runs on the 60K dev subset (20K each from Books, Handmade and Toys & Games), which lives on Sol, not in Git. `dev60k.sh` holds the paths:

| What | Default path | Built by |
|---|---|---|
| Corpus | `/scratch/jkauser/gift-concierge/data/dev_subset_v2.jsonl` | `make_dev_subset.py` |
| Image lookup | `/scratch/jkauser/gift-concierge/data/dev60k_image_lookup.json` | `build_image_lookup.py` |
| Semantic index | `/scratch/jkauser/gift-concierge/embeddings/dev60k_v2` | `embed_corpus.py` + `build_faiss_index.py` (Shashank) |
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

Cards show the image whose URL is in the lookup file written by `CODE/data/build_image_lookup.py` (`{parent_asin: image_url}`). Either copy that file to `DATA/images/image_lookup.json` (the default location) or point to it:

```bash
GC_IMAGE_LOOKUP=/scratch/jkauser/gift-concierge/data/dev60k_image_lookup.json uvicorn CODE.backend.app:app
```

The lookup must cover the corpus being served. The current script builds it for `dev_subset_v2.jsonl`; run it over the sample (or the full corpus) for those products. Products without an image show the category placeholder. "View on Amazon" links always work, since they are built from `parent_asin`.

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

## Notes and limits

- Conversations live in server memory: restarting the server, or running more than one worker, starts them over. Idle conversations are dropped after `GC_SESSION_TTL_MIN` minutes.
- `GiftConcierge` takes a dict of all products. That's fine for the sample and a 60K dev subset, but the full 27 GB corpus won't fit in memory this way; that needs a keyed product store in the LLM layer.
- Turns in one conversation run one at a time; different conversations run in parallel.
