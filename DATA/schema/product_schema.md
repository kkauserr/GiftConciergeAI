# Gift Concierge Product Schema — v2

This document defines the product interface used by downstream Gift Concierge components.

## Product Metadata

| Field | Type | Description |
|---|---|---|
| `parent_asin` | string | Amazon parent product identifier and primary join key |
| `domain` | string | Source domain: Books, Handmade Products, or Toys & Games |
| `title` | string | Product title |
| `price` | number | Original normalized product price |
| `price_num` | number | Positive numeric price used for budget filtering |
| `average_rating` | number | Amazon metadata average rating |
| `rating_number` | integer | Number of ratings associated with the product |
| `store` | string/null | Amazon store or seller/creator information when available |
| `brand_or_author` | string/null | Normalized brand or author/creator information |
| `main_category` | string/null | Broad Amazon category |
| `specific_category` | string/null | Most specific available product category |
| `categories` | array | Product category hierarchy |
| `age_min` | number/null | Minimum recommended age in years when available |
| `age_max` | number/null | Maximum recommended age in years when available |
| `features` | array | Product feature strings |
| `description` | array | Product description text |
| `details` | object | Filtered useful product details |

## Retrieval Fields

| Field | Purpose |
|---|---|
| `bm25_text` | Rich textual representation for lexical/BM25 retrieval |
| `embed_text` | Concise semantic representation for embedding/vector retrieval |

### `bm25_text`

Contains a richer combination of:

- title
- features
- description
- categories
- filtered product details

Use this field for lexical retrieval.

### `embed_text`

Contains a shorter representation emphasizing:

- title
- specific category
- category hierarchy
- brand or author
- recommended age when available
- selected product features

Use this field for semantic embeddings and vector retrieval.

## Review Fields

After review aggregation and the final metadata-review join, products may additionally contain:

| Field | Type | Description |
|---|---|---|
| `review_count` | integer | Number of reviews associated with the retained product |
| `rated_review_count` | integer | Number of reviews containing a valid rating |
| `average_review_rating` | number/null | Average calculated from rated reviews |
| `verified_review_count` | integer | Number of verified-purchase reviews |
| `helpful_votes` | integer | Total helpful votes across processed reviews |
| `review_samples` | array | Up to five structured recent review records |

Each entry in `review_samples` may contain:

- `title`
- `text`
- `rating`
- `helpful_vote`
- `verified_purchase`
- `timestamp`

## Component Usage

### Lexical Retrieval
Use `bm25_text`.

### Semantic Retrieval
Use `embed_text`.

### Hard Constraints
Use structured fields such as `price_num`, `age_min`, and `age_max`.

### Ranking
Use fields such as `average_rating`, `rating_number`, review statistics, and retrieval scores.

### Knowledge Graph / Evidence
Use structured metadata, categories, features, brand/author information, and review evidence.

## Development Sample

`DATA/samples/sample_corpus_v2.jsonl` contains 900 products:

- 300 Books
- 300 Handmade Products
- 300 Toys & Games

The sample is intended for development and integration testing. It is balanced across domains for convenience and is not intended to represent the natural distribution of the full corpus.
