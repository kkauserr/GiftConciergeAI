# Classical Baseline, Hybrid Fusion & Ranking (Disha)

This part takes a gift request (as structured constraints), finds candidate products, removes the ones that break hard rules, scores the rest and returns a ranked top-K with a score for every signal.

```
constraints ──► keyword query ──► BM25 top-200 ─┐
                                                ├─► RRF fusion ─► filters ─► 4-signal score ─► MMR ─► top-K
semantic top-200 (from LLM/retrieval part) ─────┘
```

## Files

| File | What it does |
|---|---|
| `CODE/retrieval/bm25_baseline.py` | Builds and searches the BM25 index (and TF-IDF as the simpler reference) over `bm25_text`. Also saves a small product table (`catalog.parquet`) and corpus stats used by ranking. |
| `CODE/ranking/filters.py` | Hard constraints: budget (min/max), recipient age vs `age_min`/`age_max`, exclusions. |
| `CODE/ranking/fusion.py` | Reciprocal Rank Fusion of the BM25 list and the semantic list. |
| `CODE/ranking/ranker.py` | Relevance, Bayesian-adjusted rating, log popularity and budget fit, combined by weights; then MMR for variety. |
| `CODE/ranking/pipeline.py` | Puts it all together. `GiftRanker.recommend()` is the function other parts call. |
| `CODE/ranking/ablation.py` | Turns each component off one at a time and compares the results. |
| `CODE/ranking/tune_weights.py` | Exports results for labelling, then grid-searches the weights on labelled dev queries. |
| `CODE/ranking/dev_queries.json` | 12 development queries for tuning (kept separate from the evaluation queries). |
| `CODE/data/make_dev_subset.py` | For Sol: random per-domain subset of the full corpus, small enough to download. |

## Setup

Run everything from the repo root.

```bash
pip install -r CODE/ranking/requirements.txt
python -m CODE.retrieval.bm25_baseline build                 # indexes the 900-product sample into indexes/sample
python -m pytest -q CODE/ranking/test_ranking.py
```

## Try it

```bash
python -m CODE.retrieval.bm25_baseline search "dinosaur toy" -k 5
python -m CODE.ranking.pipeline --constraints '{"recipient_age": 7, "occasion": "birthday", "interests": ["dinosaur"], "budget_max": 30}'
python -m CODE.ranking.ablation                               # writes to EVALUATIONS/results/ranking/
python -m CODE.ranking.ablation --semantic-standin           # tests the fusion path, with TF-IDF standing in for semantic retrieval
```

## Interface for the other parts

**Constraints in** (from the LLM constraint extraction). Every field is optional:

```json
{
  "recipient": "sister",
  "recipient_age": 22,
  "occasion": "graduation",
  "interests": ["coffee", "reading"],
  "keywords": [],
  "budget_min": null,
  "budget_max": 80,
  "exclusions": ["books", "mugs"]
}
```

- `interests` are counted twice in the BM25 query, `occasion` once. `recipient` is not used for search.
- `exclusions` can be a domain (`books`, `toys`, `handmade`), a broad group (`wearable`, `clothing`, `jewelry`, `food`, `decor`, expanded to concrete words in `filters.py`), or any word/phrase matched against title and categories. For anything else, the LLM should pass concrete words (e.g. "no electronics" → `["electronic", "battery", "charger"]`).
- `recipient_age` removes products whose age range doesn't fit. For recipients aged 13+, it also removes products whose title or category says kids, toddler, baby, children's, preschool etc. (skipped when the occasion mentions a baby, e.g. a baby shower).
- `keywords` are extra search words. Vague interests like "reading" match poorly in BM25 (the word appears in thousands of book descriptions), so the LLM should add concrete gift words here (e.g. `["novel", "bookmark", "book lover"]`).

**Semantic results in** (from semantic retrieval): top ~200 as `[[parent_asin, score], ...]` or `[{"parent_asin": ..., "score": ...}, ...]`, best first.

**Ranked results out** (to explanation and evaluation):

```python
from CODE.ranking.pipeline import GiftRanker, RankConfig
ranker = GiftRanker("indexes/sample")
out = ranker.recommend(constraints, semantic_results=semantic_list)
```

```json
{
  "query": "coffee reading coffee reading graduation",
  "n_after_filters": 143,
  "dropped": {"over_budget": 40, "excluded": 12},
  "latency_ms": 31.2,
  "results": [
    {"rank": 1, "parent_asin": "B0...", "title": "...", "domain": "Handmade Products", "price": 34.0,
     "average_rating": 4.8, "rating_number": 512, "final_score": 0.871,
     "signals": {"relevance": 0.94, "quality": 0.91, "popularity": 0.72, "budget_fit": 0.88},
     "sources": {"bm25_rank": 3, "semantic_rank": 1}}
  ]
}
```

## How the score works

- **Relevance**: retrieval score (BM25, or the RRF score when fused), scaled 0–1 within the candidate set. Items below 0.25 are dropped, so a well-rated but off-topic product can't climb the list.
- **Quality**: `(n * avg + m * C) / (n + m)` with C = corpus mean rating and m = median rating count. One 5-star rating counts for less than thousands of 4.7s.
- **Popularity**: `log(1 + rating_count)` divided by the 99th-percentile value.
- **Budget fit**: highest when price is about 70% of `budget_max` (or the middle of the range if `budget_min` is given). Neutral when there's no budget.
- **Default weights** (tuned on the 12 labelled dev queries, 900-product sample): relevance 0.7, quality 0.1, popularity 0.0, budget fit 0.2. Re-tune on the larger subset.
- **MMR** (λ = 0.9, tuned) re-orders the top 50. Item similarity = 0.6 text cosine + 0.25 same category + 0.15 same domain.

## Tuning the weights

```bash
python -m CODE.ranking.tune_weights pool                      # writes CODE/ranking/dev_labels.csv
# fill the grade column: 2 = great gift, 1 = okay, 0 = not relevant
python -m CODE.ranking.tune_weights tune
```

## Current results (900-product sample, 12 dev queries, tuned weights)

| Config | Constraint satisfaction | nDCG@10 | P@10 | Intra-list similarity |
|---|---:|---:|---:|---:|
| BM25 only | 0.817 | 0.654 | 0.592 | 0.283 |
| TF-IDF only | 0.775 | 0.583 | 0.525 | 0.274 |
| BM25 + filters | 1.000 | 0.709 | 0.633 | 0.262 |
| Full (filters + ranking + MMR) | 1.000 | 0.737 | 0.633 | 0.258 |
| Full without MMR | 1.000 | 0.756 | 0.658 | 0.273 |
| Full + per-interest search | 1.000 | 0.662 | 0.592 | 0.258 |

Per-interest search (`split_interests`) runs one BM25 search per interest and merges them. It was tried to stop one interest drowning out another, but it hurt, because single-word interests are weak BM25 queries. It's off by default. MMR costs some nDCG but makes the list more varied. nDCG doesn't reward variety, so this trade-off is expected. These are development numbers only: the sample is small and the same queries were used for tuning.

## Full corpus (on Sol)

The 900-row sample is the first 300 rows of each domain, so it's fine for testing the code but not for judging quality. To get something bigger, someone with Sol access runs:

```bash
python CODE/data/make_dev_subset.py --per-domain 20000 --output dev_subset_v2.jsonl   # ~190 MB, 60K products
python -m CODE.retrieval.bm25_baseline build --corpus dev_subset_v2.jsonl --out indexes/dev60k
```

For the full 3.78M products, run the same `build` on Sol with plenty of memory (64 GB+). Add `--skip-tfidf` if TF-IDF isn't needed at full scale.
