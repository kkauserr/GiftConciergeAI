# Gift Concierge: Code Interfaces

## Semantic Retrieval + LLM / Conversation Layer

Owner: Shashank G. Code: `CODE/retrieval/` (embeddings, FAISS, search) and `CODE/llm/` (constraint extraction, dialogue state, clarification, RAG explanations, demo). Tests: `CODE/tests/`.

```
user message
  -> constraint extraction (LLM, JSON schema)      llm/constraints.py
  -> dialogue state + conflict rules               llm/dialogue_state.py
  -> one clarifying question if needed             llm/clarification.py
  -> query text -> semantic top-100                retrieval/semantic_search.py
  -> BM25 / RRF / filtering / ranking   (Disha)    llm/stub_ranker.py until ready
  -> RAG explanation (+ KG facts, Jenna)           llm/explanation.py
```

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in FOUNDRY_ENDPOINT / FOUNDRY_API_KEY / FOUNDRY_DEPLOYMENT
python CODE/retrieval/embed_corpus.py         # sample -> DATA/embeddings/sample_v2/ (gitignored)
python CODE/retrieval/build_faiss_index.py
python CODE/llm/demo_cli.py                   # or --offline to run without credentials
python -m pytest CODE/tests
```

`.env`, `DATA/embeddings/`, `*.faiss` and `*.npy` are gitignored. Don't commit index or embedding files.

### Full corpus on Sol

Both scripts can resume. If a job hits its time limit, resubmit the same command and it continues from the last finished shard.

```bash
python CODE/retrieval/embed_corpus.py \
    --input /scratch/jkauser/gift-concierge/data/final_product_corpus_v2.jsonl \
    --out-dir /scratch/$USER/gift-concierge/embeddings/full_v2 \
    --batch-size 512 --device cuda --fp16
python CODE/retrieval/build_faiss_index.py --index-dir /scratch/$USER/gift-concierge/embeddings/full_v2
```

The index directory contains `index.faiss` (IndexFlatIP, about 5.8 GB for 3.78M × 384 float32), `row_to_asin.txt` (line *i* is the `parent_asin` of FAISS row *i*), `manifest.json`, and the shard files.

Model: `BAAI/bge-small-en-v1.5`. Vectors are L2-normalized, so the inner-product score equals cosine similarity. Queries get the BGE prefix `"Represent this sentence for searching relevant passages: "`. Documents (`embed_text`) don't.

---

### For Disha: retrieval output and constraints JSON

**Semantic results** come from `SemanticRetriever(index_dir).search(query_text, top_n=100)`:

```json
[
  {"parent_asin": "B07NDY3VVR", "score": 0.681234, "rank": 1},
  {"parent_asin": "B00EYHD42G", "score": 0.674101, "rank": 2}
]
```

- `rank` starts at 1, and the list is sorted by `score` from highest to lowest.
- `score` is cosine similarity in [-1, 1].
- `search_batch([q1, q2], top_n)` returns one such list per query.

**Constraints JSON** comes from `GiftConstraints.to_handoff()`. Every key is always present:

```json
{
  "recipient": "my 8 year old nephew",
  "occasion": "birthday",
  "interests": ["dinosaurs", "space"],
  "exclusions": ["puzzle"],
  "budget_min": null,
  "budget_max": 22.5,
  "budget_flexible": false,
  "age": 8.0,
  "age_group": null,
  "category_preference": "Toys & Games",
  "target_age_range": [8.0, 8.0]
}
```

- `category_preference` is one of `"Books"`, `"Handmade Products"`, `"Toys & Games"`, or `null`. It matches the product `domain` field.
- `age_group` is one of `baby`, `toddler`, `child`, `teen`, `adult`, `senior`, or `null`.
- `target_age_range` is `[min, max]` in years, with `null` meaning no bound. It comes from `age`, or else from `age_group`. Compare it to the product `age_min`/`age_max`.
- `budget_flexible: true` means the user explicitly said budget doesn't matter.
- Exclusions are lowercase free text, such as "batteries" or "plush toys". Apply them as filters on title, categories and features.
- The retrieval query text leaves out budget and exclusions on purpose. They're for your hard filters.

**The interface your ranker should expose** (the conversation loop calls it like this):

```python
rank(retrieved, constraints, products, top_k=5) -> [{"parent_asin", "score", "rank"}]
#   retrieved   semantic results above
#   constraints the constraints JSON above
#   products    {parent_asin: product record}
```

Pass it in with `GiftConcierge(..., ranker=your_rank)`. The placeholder is `CODE/llm/stub_ranker.py`. It applies price, category, exclusion and age filters, then sorts by semantic score plus a small rating boost.

---

### For Jenna: KG facts in explanations

`explain_recommendations(..., kg_facts=...)` accepts either of these:

- a dict: `{parent_asin: [fact, ...]}`
- any object with `get_facts(parent_asin) -> [fact, ...]`

A fact can be a string, `{"subject": ..., "relation": ..., "object": ...}`, or `{"text": ...}`. Up to 5 facts per product go into the prompt as evidence lines `P1.kg[0]`, `P1.kg[1]`, and so on. When the model relies on a fact, it shows up in `used_evidence` with `"field": "kg"`.

---

### For Shashank V: turn responses and faithfulness evidence

`GiftConcierge(llm, retriever, products).handle(message)` returns a dict.

| key | when | content |
|---|---|---|
| `type` | always | `"question"` or `"recommendations"` |
| `message` | always | text shown to the user |
| `slot` | questions | `interests`, `budget`, `conflict:<kind>`, `rephrase`, `no_results` |
| `turn`, `constraints` | always | turn number and the constraints JSON |
| `state_changes` | always | `[{turn, field, old, new, rule, utterance, timestamp}]` |
| `conflicts` | always | `[{kind, fields, question, details}]` |
| `extraction_error` | always | `null`, or the error if extraction failed |
| `query_text`, `retrieved`, `ranked` | recommendations | query, semantic top-N, final ranking |
| `explanations`, `evidence` | recommendations | see below |

Each explanation:

```json
{
  "parent_asin": "B07NDY3VVR",
  "rank": 1,
  "explanation": "A LEGO Creator set with ...",
  "evidence_ids": ["P1.title", "P1.feature[0]", "P1.rating"],
  "used_evidence": [
    {"id": "P1.feature[0]", "parent_asin": "B07NDY3VVR", "field": "features", "text": "..."}
  ],
  "invalid_citations": [],
  "fallback": false
}
```

- `evidence` maps each product ID (`P1`, `P2`, ...) to all the evidence lines in the prompt. `used_evidence` is the subset the model cited.
- Check faithfulness of `explanation` against `used_evidence` (or against the full `evidence` for that product).
- `invalid_citations` lists IDs the model cited that weren't in the prompt for that product. These are removed before `used_evidence` is built.
- `fallback: true` means the LLM produced no explanation, so a metadata template (title, price, rating) was used instead. Leave these out of LLM faithfulness scores.

Evidence IDs per product: `title`, `price`, `rating`, `category`, `age`, `brand`, `feature[j]` (up to 4), `description[0]` (only when there are no features), `review_stats`, `review[j]` (up to 3, most helpful first), `kg[j]`.

> **Note:** `DATA/samples/sample_corpus_v2.jsonl` comes from the corpus before the review join, so it has no review fields. On the sample, explanations cite metadata only. Review evidence shows up once we use a sample from `final_product_corpus_v2.jsonl`.

---

### Multi-turn conflict rules

The rules are implemented in `CODE/llm/dialogue_state.py` and tested in `CODE/tests/test_dialogue_state.py`. The LLM only extracts what each message says. Python decides how that changes the state.

| User says | Rule |
|---|---|
| "actually $150" after "under $50" | The newer explicit value overrides: `budget_max` 50 → 150 |
| a new bound makes an old one impossible | The older bound is cleared (`cleared_inconsistent_older_value`) |
| "cheaper" / "more expensive" | Existing bounds × 0.75 / × 1.33 |
| "$20 cheaper" | Existing bounds − $20 |
| "cheaper" with no budget yet | Ask for a price range |
| new interests | Added to the list |
| "... instead" | Replaces the list. Dropped interests are **not** excluded |
| "she's not into puzzles anymore" | Removed from interests and added to exclusions |
| re-adding something excluded | The exclusion is removed (the newer statement wins) |
| "budget doesn't matter" | Budget cleared and `budget_flexible = true` |
| a message contradicts itself ("at least $80 but under $50", "likes horses, no horses", "a book, no books", age 30 + toddler) | Affected fields stay unchanged, the rest of the message is applied, and **one question is asked** |

Every change is recorded in `state.history` and appended to `logs/state_changes.jsonl` (gitignored) along with the rule name and the utterance that triggered it.

### Clarification

At most one question per turn. The order is: a contradiction from this turn, then missing interests, then a missing budget (unless the user said "no limit"). Each missing slot is asked about once. If the user skips it, we recommend with what we have.

### LLM provider

All calls go through `CODE/llm/llm_client.py`:

- `LLMClient.complete(...)` returns text.
- `LLMClient.complete_json(..., schema, name)` uses strict JSON-schema output.

This runs on Microsoft Foundry through `AzureOpenAI`. To switch providers, implement those two methods in another class.

`FakeLLMClient` is for tests. `offline_llm.OfflineLLM` is a regex stand-in for the demo only and must not be used for evaluation.
