# Gift Concierge Dataset

## Overview

Gift Concierge uses product metadata and customer review data from the Amazon Reviews 2023 dataset released by McAuley Lab.

The dataset supports lexical retrieval, semantic retrieval, constraint filtering, ranking, knowledge-graph construction, and evidence-grounded gift recommendations.

Because the complete raw and processed datasets are large, they are not stored in this GitHub repository. The repository instead contains the preprocessing code, schema documentation, and a small development sample.

---

## Data Source

**Dataset:** Amazon Reviews 2023  
**Source:** McAuley Lab

Nine candidate Amazon product domains were initially evaluated for field completeness and suitability for the recommendation pipeline.

The final three domains were selected based on usable metadata coverage:

- Books
- Handmade Products
- Toys & Games

These domains also provide a diverse mix of interest-based, personalized, recreational, and family-oriented gift products.

---

## V2 Product Metadata Statistics

| Domain | Raw Products | Retained Products | Retention |
|---|---:|---:|---:|
| Books | 4,448,181 | 3,266,409 | 73.43% |
| Handmade Products | 164,817 | 94,173 | 57.14% |
| Toys & Games | 890,874 | 416,179 | 46.72% |
| **Total** | **5,503,872** | **3,776,761** | **68.62%** |

The v2 preprocessing pipeline applies stricter numeric price validation than v1 and retains only products suitable for downstream constraint-aware retrieval.

---

## Review Statistics

Reviews from the same three Amazon domains are joined to retained products using `parent_asin`.

| Domain | Reviews Processed | Products with Review Evidence |
|---|---:|---:|
| Books | 29,475,453 | 3,264,872 |
| Handmade Products | 664,162 | 94,124 |
| Toys & Games | 16,264,812 | 416,078 |
| **Total** | **46,404,427** | **3,775,074** |

Final review coverage:

- Final products: **3,776,761**
- Products with review evidence: **3,775,074**
- Products without review evidence: **1,687**
- Review coverage: **99.96%**

---

## Product Preprocessing

The v2 pipeline validates and standardizes product metadata before retrieval.

Core validation includes:

- valid `parent_asin`
- non-empty title
- positive numeric price
- available product rating
- usable descriptive content

The preprocessing pipeline additionally extracts or normalizes:

- numeric price
- store information
- brand or author
- broad and specific categories
- recommended age information when available
- product features
- descriptions
- selected useful product details

Products are not filtered using a minimum `rating_number` threshold during preprocessing. Such thresholds can instead be applied during retrieval or ranking experiments.

---

## Retrieval Representations

V2 maintains separate textual representations for lexical and semantic retrieval.

### `bm25_text`

A richer representation containing:

- title
- features
- description
- categories
- filtered product details

This field is intended for lexical/BM25 retrieval.

### `embed_text`

A more concise semantic representation emphasizing:

- title
- specific category
- category hierarchy
- brand or author
- recommended age when available
- selected product features

This field is intended for embedding generation and vector retrieval.

Keeping these representations separate allows each retrieval method to use text appropriate to its strengths.

---

## Review Processing

Review records are matched to retained products using `parent_asin`.

For each retained product, the pipeline aggregates:

- `review_count`
- `rated_review_count`
- `average_review_rating`
- `verified_review_count`
- `helpful_votes`
- up to five recent structured review samples

Each review sample can contain:

- title
- review text
- rating
- helpful votes
- verified-purchase status
- timestamp

Review text is cleaned before storage, and the average review rating is calculated only from reviews containing valid ratings.

---

## Final V2 Corpus

The metadata and review summaries are joined into:

`final_product_corpus_v2.jsonl`

Confirmed final statistics:

- **3,776,761 products**
- **3,775,074 products with review evidence**
- **99.96% review coverage**
- approximately **27 GB**

The complete corpus is maintained on ASU Sol and is intentionally excluded from GitHub.

Detailed field documentation is available in:

`DATA/schema/product_schema.md`

---

## Development Sample

The repository includes:

`DATA/samples/sample_corpus_v2.jsonl`

This contains **900 products**:

- 300 Books
- 300 Handmade Products
- 300 Toys & Games

The sample is intended for development and component integration so retrieval, ranking, LLM, and knowledge-graph code can be developed without loading the full corpus.

It is deliberately balanced across the three domains and should not be interpreted as a statistically representative sample of the full corpus.

---

## Preprocessing Pipeline

The v2 data pipeline is:

Amazon Reviews 2023 raw metadata

→ metadata validation and normalization

→ BM25 and embedding text construction

→ retained product corpus

→ review filtering and aggregation by `parent_asin`

→ metadata-review join

→ final v2 product corpus

The implementation is located in:

`CODE/data/`

Main scripts:

- `analyze_metadata.py` — evaluates category metadata completeness
- `analyze_prices.py` — analyzes raw price quality and formats
- `preprocess_products.py` — validates and standardizes product metadata and constructs retrieval representations
- `process_reviews.py` — aggregates review evidence for retained products
- `build_final_corpus.py` — joins product metadata with review summaries
- `create_sample_corpus.py` — creates the balanced development sample

---

## Downstream Component Usage

The processed corpus provides a shared interface for the remaining Gift Concierge components.

**Lexical retrieval:** use `bm25_text`.

**Semantic retrieval:** use `embed_text`.

**Hard constraint filtering:** use structured fields such as `price_num`, `age_min`, and `age_max`.

**Ranking:** use retrieval scores together with metadata rating and review statistics.

**Knowledge graph and explanation:** use structured product metadata and review evidence.

The master corpus retains products that pass data-quality requirements. Query-specific constraints and experimental thresholds should be applied by downstream retrieval and ranking components rather than permanently removing those products during preprocessing.

---

## Repository Data Policy

Large raw and processed datasets are intentionally excluded from version control.

GitHub stores:

- preprocessing code
- schema documentation
- small development samples
- retrieval and ranking code
- LLM and knowledge-graph components
- evaluation code and results

The complete Amazon source files and full processed corpus remain on ASU Sol.
