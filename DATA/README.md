# Gift Concierge Dataset

## Overview

Gift Concierge uses product metadata and customer review data from the Amazon Reviews 2023 dataset released by McAuley Lab.

The dataset is used to support product retrieval, semantic search, ranking, knowledge-graph construction, and evidence-grounded gift recommendations.

Because the complete raw and processed datasets are large, they are not stored directly in this GitHub repository. Instead, this repository contains the preprocessing code required to construct the processed corpus from the original public dataset.

---

## Data Source

**Dataset:** Amazon Reviews 2023  
**Source:** McAuley Lab

The original dataset contains product metadata and customer reviews across multiple Amazon product domains.

For Gift Concierge, nine candidate product domains were initially evaluated for data quality and suitability for gift recommendation. The final three domains were selected based on the availability of the fields required by the recommendation system:

- Books
- Handmade Products
- Toys & Games

These categories provide strong coverage of price, rating, and descriptive product information while also representing diverse gift types.

---

## Dataset Statistics

### Product Metadata

| Domain | Raw Products | Retained Products | Retention |
|---|---:|---:|---:|
| Books | 4,448,181 | 3,526,180 | 79.27% |
| Handmade Products | 164,817 | 94,173 | 57.14% |
| Toys & Games | 890,874 | 416,179 | 46.72% |
| **Total** | **5,503,872** | **4,036,532** | **73.34%** |

The final processed product corpus contains **4,036,532 products**.

### Review Data

Reviews from the same three Amazon domains were processed and joined with the retained products.

- Raw reviews processed: **46,404,096**
- Products in final corpus: **4,036,532**
- Products with review evidence: **4,034,775**
- Products without review evidence: **1,757**
- Review coverage: **99.96%**

---

## Product Preprocessing

Raw product metadata is filtered to retain products containing the fields necessary for retrieval and recommendation.

A product must contain:

- `parent_asin`
- `title`
- `price`
- `average_rating`
- at least one descriptive field: `features` or `description`

Additional descriptive information is retained when available.

The searchable `product_text` representation is constructed by combining:

- title
- features
- description
- categories
- product details

This provides a unified textual representation that can be used by lexical and semantic retrieval methods.

---

## Review Processing

Amazon review records are linked to product metadata using `parent_asin`.

Reviews are aggregated at the product level to produce:

- `review_count`
- `average_review_rating`
- `verified_review_count`
- `helpful_votes`
- `review_samples`

This review evidence can be used by downstream ranking, knowledge-graph, and explanation components of Gift Concierge.

---

## Final Corpus Schema

Each record in the final processed corpus contains:

| Field | Description |
|---|---|
| `parent_asin` | Amazon parent product identifier |
| `domain` | Product domain |
| `title` | Product title |
| `price` | Product price |
| `average_rating` | Metadata average rating |
| `rating_number` | Number of product ratings |
| `product_text` | Combined searchable product representation |
| `review_count` | Number of processed reviews |
| `average_review_rating` | Average rating calculated from review records |
| `verified_review_count` | Number of verified-purchase reviews |
| `helpful_votes` | Total helpful votes across reviews |
| `review_samples` | Representative review text samples |

---

## Preprocessing Pipeline

The complete data pipeline follows:

Amazon Reviews 2023 raw data

→ Product metadata validation and filtering

→ Product text construction

→ Review aggregation by `parent_asin`

→ Metadata and review join

→ Final standardized product corpus

The preprocessing implementation is located in:

`CODE/data/`

The main scripts are:

- `analyze_metadata.py` — evaluates metadata completeness and category feasibility
- `preprocess_products.py` — filters and standardizes product metadata and constructs `product_text`
- `process_reviews.py` — aggregates review information by product
- `build_final_corpus.py` — joins processed metadata with review aggregates and produces the final corpus

---

## Reproducing the Dataset

The complete processed corpus is not committed to GitHub because of its size.

To reproduce the dataset, obtain the Amazon Reviews 2023 metadata and review files for:

1. Books
2. Handmade Products
3. Toys & Games

Then run the preprocessing scripts in `CODE/data/` in the following order:

1. `preprocess_products.py`
2. `process_reviews.py`
3. `build_final_corpus.py`

The scripts transform the original Amazon Reviews 2023 files into the standardized corpus used by Gift Concierge.

The resulting dataset is:

`final_product_corpus.jsonl`

Final corpus statistics:

- **4,036,532 products**
- **4,034,775 products with review evidence**
- **99.96% review coverage**
- approximately **15 GB**

The full corpus is maintained on ASU Sol for project development and large-scale retrieval experiments.

---

## Repository Data Policy

Large raw and processed data files are intentionally excluded from version control.

GitHub is used to store:

- preprocessing code
- documentation
- small example/sample data
- retrieval and ranking code
- evaluation code and results

Large Amazon source files and the complete processed corpus remain outside the repository.

This separation keeps the repository lightweight while allowing the dataset construction process to remain documented and reproducible.
