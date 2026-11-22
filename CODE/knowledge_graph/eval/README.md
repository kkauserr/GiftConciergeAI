# Manual evaluation of inferred KG relationships

No precision or recall has been measured yet. All confidence values in the graph are heuristic rule scores, not probabilities. Report numbers only after the sheets below have been labeled.

## Files

| File | What it is | Edit it? |
|---|---|---|
| `labeling_sheet_v1.csv` | 60 products (20 per domain: 10 with a predicted interest, 10 without). Its `predicted_*` columns were written with `interests@1.0.0` / `aspects@1.0.0`. | **Yes**: fill in the annotator columns. Never regenerate it. |
| `labeling_sheet_v1_predictions_current.csv` | The *current* predictions (interests with `match_type`, attribute aspects, review-supported aspects, review concerns) for the same 60 products. | No. Regenerate it after any vocabulary change. |
| `review_evidence_sheet_v1.csv` | 107 review-evidence items (`SUPPORTS`), stratified by aspect and polarity: up to 4 per (aspect, polarity) cell, so rare negative and neutral items are included. | **Yes**: fill in the annotator columns. |

Label from the product (title, category path, product page if needed) and the review text, **not** from the predictions.

All writers refuse to overwrite an existing file. To make new sheets, from `CODE/`:

```
python -m knowledge_graph.build_kg --labeling-sheet knowledge_graph/eval/labeling_sheet_v2.csv
python -m knowledge_graph.build_kg --review-sheet knowledge_graph/eval/review_evidence_sheet_v2.csv
python -c "from knowledge_graph.graph import KnowledgeGraph, iter_jsonl; from knowledge_graph.reports import write_sheet_predictions; kg = KnowledgeGraph.build(iter_jsonl('../DATA/samples/sample_corpus_v2.jsonl')); write_sheet_predictions(kg, 'knowledge_graph/eval/labeling_sheet_v1.csv', 'knowledge_graph/eval/labeling_sheet_v1_predictions_new.csv')"
```

## 1. Interests (MATCHES)

Use `labeling_sheet_v1.csv`. Fill in:

| Column | What to enter |
|---|---|
| `gold_interests` | Every interest label (from `../vocab/README.md`) that a gift recipient with that interest would plausibly want this product for, separated by `;`. Leave empty if none apply. |
| `wrong_predicted_interests` / `missed_interests` | Optional shortcuts. `gold_interests` is what gets scored. |
| `notes` | Use `primary`, `themed` or `incidental` for borderline cases (see below). |

Guidelines:

- **An interest means the product suits someone with that interest.** Product type alone doesn't count: a plain silver necklace has no interest, but a dachshund necklace matches `dogs`.
- **Primary**: the product is about or for the interest (a fishing lure, a history book, a chess set). Always gold.
- **Themed**: the product carries the interest as a motif or license (a dog-shaped piggy bank, a Marvel figure). Gold if a fan of that interest would plausibly want it.
- **Incidental**: the interest appears only in passing (a tea set inside an anime figure, "fits Yugioh cards" on a football binder, a cross-sell line). Not gold.
- **Party supplies and nursery decor**: the motif is for an event or a baby's room. Not gold, unless the category itself is the interest.
- **Never label recipients or occasions** (mom, teacher, birthday, Halloween). They are not interests.

Score against `labeling_sheet_v1_predictions_current.csv` (join on `sheet_id`). Do not score against the v1.0 columns in the sheet itself, because they predate the v1.1 false-positive fixes.

Metrics:

- **Precision** = correct predicted labels ÷ all predicted labels. Report it per domain, per evidence source (category, Theme, title; feature text never stands alone in v1.1) and per `match_type`.
- **Recall (rough)** = correct predicted ÷ (correct predicted + gold labels not predicted). The 30 "no prediction" products give the main signal for missed interests.
- Targets from the plan: ≥ 0.85 precision for category rules and ≥ 0.70 for title rules. Rules below target get revised or removed in the next vocabulary version.

## 2. Attribute aspects (listing HAS_ASPECT)

Same sheet, columns `wrong_predicted_aspects` and `missed_aspects`.

- Judge only objective facts: material, personalizable, handcrafted, battery powered and so on. Ignore quality claims like "durable".
- A material counts only if **the product itself** is made of it. Packaging, instructions, finishes and kit components don't count: "1 plastic funnel" in a chemistry kit is not a plastic product.
- Listing-stated aspects (`evidence_type = listing_stated`) are seller statements. Mark them wrong only if the listing contradicts itself or the attribute plainly belongs to something else. You can't verify the physical product.
- **Precision** per `evidence_type` (structured vs listing_stated) and per aspect.

## 3. Review evidence (SUPPORTS and review-derived HAS_ASPECT)

Use `review_evidence_sheet_v1.csv`. Each row is one evidence item: one clause of one review, linked to one quality aspect. Read `full_review_text`, then fill in:

| Column | What to enter |
|---|---|
| `gold_aspect_correct` | `yes` if the clause is about this quality aspect **of this product**, otherwise `no`. Examples of `no`: "the box was crushed" for durability, "the author's other book is well worth reading" for value, a comment about the seller. |
| `gold_polarity` | What the clause says about the aspect: `positive`, `negative` or `neutral`. Neutral covers hopes, questions, conditionals, "not flimsy" and similar. Leave it empty if `gold_aspect_correct` is `no`. |
| `notes`, `annotator` | Comments and your initials. |

Guidelines:

- **Polarity comes from the text, never from the star rating.** A 1-star review saying "very sturdy" is positive about durability. The rating explains why the row may be flagged `rating_conflict`.
- **Negative evidence is never support.** "Broke after a week" is negative for durability, not "a review about durability".
- Sarcasm counts as what the reviewer meant.

Metrics, over rows with gold labels:

- **Aspect precision** = rows with `gold_aspect_correct = yes` ÷ all rows, per aspect.
- **Polarity accuracy** = rows where `gold_polarity` equals `predicted_item_polarity` ÷ rows with `gold_aspect_correct = yes`, per predicted polarity. Watch **positive → negative** and **negative → positive** errors closely: they are the costly ones.
- The sheet is stratified, so do not read its polarity mix as the corpus distribution. Use the counts from `build_kg --coverage` for that.

Review-derived HAS_ASPECT (rows with `in_review_aspect_has_aspect = True`, plus the `review_aspects` column in the predictions file): for each product-level claim, read all of the product's reviews (`KnowledgeGraph.review_evidence(asin, aspect)`) and judge whether "reviewers say this product has good <aspect>" is a fair summary. Report the share judged fair.

## Recording results

Save labeled copies under new names (e.g. `labeling_sheet_v1_labeled_<initials>.csv`) and record the vocabulary versions you scored against (`vocab_versions` column of the predictions file). With two annotators, label 20 shared rows and report raw agreement before splitting the rest.
