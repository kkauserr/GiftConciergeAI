# Knowledge Graph integration guide (`get_evidence` contract `kg-evidence/1.0`)

This guide is for the conversational / LLM component. It covers everything needed to use the knowledge graph (KG): you don't need to know how the graph is stored (NetworkX) or how its rules work.

**What the KG provides.** Given candidate products (from your retriever), `get_evidence()` returns structured, traceable evidence for each one:
- brand or author, and category;
- matched gift interests from a shared 31-label vocabulary;
- objective attributes (materials, personalizable, handcrafted…);
- review-supported qualities and negative review concerns, with quoted review snippets;
- a ready-made list of claims that are safe to state, and cautions.

It doesn't retrieve, rank or generate text. Those stay in your component.

- **Public module:** `knowledge_graph.api`. Nothing else in `knowledge_graph/` is part of the contract.
- **Response schema:** [`contract/get_evidence.schema.json`](contract/get_evidence.schema.json), a JSON Schema checked by the tests on every product in the sample.
- **Contract version:** `kg-evidence/1.0`, returned as `contract_version` in every response.
- **Vocabulary versions:** `interests@1.1.0`, `aspects@1.2.0`, returned as `vocabulary`.
- **No LLM dependency:** the KG never calls a model, and responses are plain JSON.

## 1. Load once, query many times

```python
from knowledge_graph.api import load_kg

kg = load_kg()          # builds the graph from DATA/samples/sample_corpus_v2.jsonl (~15 s), then caches it
```

`load_kg()` returns the same object on every later call in the process, so you can call it from anywhere. `get_evidence()` never rebuilds the graph: a call for a handful of products takes a few milliseconds.

| Situation | Call |
|---|---|
| Default 900-product sample | `load_kg()` |
| Another corpus JSONL file | `load_kg("path/to/corpus.jsonl")` |
| A prebuilt graph (recommended for the full corpus) | `load_kg(export_dir="kg_full")` |
| Force a rebuild | `load_kg(..., reload=True)` |
| Explicit object, no global cache | `KGService.from_jsonl(path)`, `KGService.from_export(dir)` |

Build an export once, from `CODE/`:

```
python -m knowledge_graph.build_kg path/to/corpus.jsonl --out kg_full
```

Loading the sample export takes about 0.5 s instead of 15 s, and the answers are identical (tested). Don't commit export directories: they are generated output.

Requirements: Python 3.10+ and `networkx>=3.0`. Install from `CODE/`:

```
pip install -r knowledge_graph/requirements.txt
```

Run your code with `CODE/` on the import path (e.g. start Python from `CODE/`, or add it to `PYTHONPATH`), so that `import knowledge_graph` works.

**Not yet available:** a full-corpus graph (about 3.78M products, on ASU Sol). Until it is built and its export location agreed, use the 900-product sample. Your code shouldn't need to change: only the `load_kg(export_dir=...)` argument will.

To check your setup, from `CODE/`:

```
python -m unittest discover -s knowledge_graph/tests -t . -v      # 215 tests
python -m knowledge_graph.examples.get_evidence_demo              # end-to-end demo, no LLM
```

## 2. Minimal working example

```python
import json
from knowledge_graph.api import load_kg, evidence_to_text

kg = load_kg()

# 1. Your retriever returns candidate products, best first.
candidates = ["B0164RRP14", "B09XDX9Q8T", "NOT_A_REAL_ASIN"]
# 2. Your constraint extraction returns interest labels from interests.json.
interests = ["fishing_hunting"]

resp = kg.get_evidence(candidates, query_interests=interests, max_reviews=2)

for w in resp["warnings"]:
    print("KG warning:", w["code"], w["value"])

context = "\n\n".join(evidence_to_text(r) for r in resp["results"])   # grounded prompt context
print(context)
print(json.dumps(resp)[:200])                                           # it is plain JSON
```

The same code runs as a demo, with no LLM needed:

```
python -m knowledge_graph.examples.get_evidence_demo
python -m knowledge_graph.examples.get_evidence_demo --asins B0164RRP14 1683239822 --interests dogs --json
```

`kg.get_evidence(...)` is the main call. A module-level `knowledge_graph.api.get_evidence(...)` does the same on the default graph.

## 3. Input

```python
get_evidence(parent_asins, query_interests=None, max_reviews=3)
```

| Argument | Type | Rules |
|---|---|---|
| `parent_asins` | list of str | Required: **parent** ASINs (the corpus `parent_asin` field), not variant/child ASINs. A child ASIN is not in the KG and comes back as `found: false`. Pass them in the order you want results back. **Results keep this order exactly**, including duplicates. An empty list returns `results: []`. Passing a single string instead of a list raises `TypeError`. |
| `query_interests` | list of str or `None` | Optional labels from `vocab/interests.json`. See below. Passing a single string raises `TypeError`. |
| `max_reviews` | int, 0–10 | The most individual reviews quoted **per aspect** (in `review_observations`, `supporting_reviews` and concern `examples`). Default 3. Values above 10 are capped, with a warning. A negative number or non-integer raises `ValueError`. |

### Interest labels (`interests@1.1.0`)

The 31 accepted labels follow. The aliases are example user phrasings for your extraction prompt, **not** accepted inputs. `kg.interest_vocabulary()` returns labels, display names, descriptions and all aliases at runtime, and `kg.interest_labels()` returns the plain list. Building your JSON-schema enum from `kg.interest_labels()` keeps the two components in sync automatically.

| Label | Display name | Example user phrasings |
|---|---|---|
| `history` | History | history buff, loves history, military history |
| `mystery_thriller` | Mystery & Thriller | mystery novels, thrillers, true crime |
| `scifi_fantasy` | Sci-Fi & Fantasy | sci-fi fan, fantasy fan, dragons |
| `romance` | Romance | romance novels, love stories |
| `poetry_literature` | Poetry & Literature | poetry, classic literature, literary fiction |
| `biography_memoir` | Biography & Memoir | biographies, memoirs, true stories |
| `horror_spooky` | Horror | horror fan, scary stories, stephen king |
| `humor` | Humor | funny, jokes, pranks |
| `faith_spirituality` | Faith & Spirituality | christian, religious, faith |
| `science_tech` | Science & Technology | science, stem, engineer |
| `space_astronomy` | Space & Astronomy | space, astronomy, planets |
| `dinosaurs` | Dinosaurs | dinosaurs, dino, t-rex |
| `animals_wildlife` | Animals & Wildlife | animal lover, wildlife, birds |
| `dogs` | Dogs | dog lover, dog mom, dog dad |
| `cats` | Cats | cat lover, cat mom, cat lady |
| `cooking_baking` | Cooking & Baking | cooking, baking, home chef |
| `coffee_tea` | Coffee & Tea | coffee lover, tea lover, caffeine |
| `gardening_plants` | Gardening & Plants | gardener, gardening, plant lover |
| `outdoors_camping` | Outdoors & Camping | camping, hiking, outdoorsy |
| `fishing_hunting` | Fishing & Hunting | fishing, fisherman, angler |
| `sports_fitness` | Sports & Fitness | sports fan, athlete, soccer |
| `travel` | Travel | traveler, travel lover, wanderlust |
| `music` | Music | musician, music lover, guitarist |
| `art_drawing` | Art & Drawing | artist, painter, drawing |
| `photography` | Photography | photographer, camera, photography hobby |
| `crafts_diy` | Crafts & DIY | crafter, knitter, sewing |
| `tabletop_games` | Board & Card Games | board games, card games, game night |
| `puzzles` | Puzzles | puzzles, jigsaw, crosswords |
| `vehicles_cars` | Vehicles & Cars | car guy, car enthusiast, trucks |
| `superheroes_comics` | Superheroes & Comics | marvel fan, dc fan, superheroes |
| `anime_manga` | Anime & Manga | anime fan, manga, otaku |

How labels are matched:
- **Normalization.** Case, surrounding spaces, spaces and hyphens are normalized: `"Fishing Hunting"` and `"fishing-hunting"` become `fishing_hunting`, with a `normalized_interest` warning.
- **Unknown labels** (`"ponies"`, `"dog lover"`, numbers) are dropped with an `unknown_interest` warning. They never cause an error.
- **Only exact labels count.** Aliases are not resolved, so your extractor must output labels.
- **Interests never filter or re-rank products.** They only set `matches_query` and `query_interest_match`. A product with no interest match is still a valid candidate; many good gifts have none. Ranking is your component's decision.
- **Out of scope for interests:** recipients (mom, teacher), occasions (birthday, Halloween), age and budget. They belong in your conversation state, not in `query_interests`. Specific franchises ("Star Wars") belong in your free-text retrieval query.

## 4. Response

Top level:

```json
{
  "contract_version": "kg-evidence/1.0",
  "vocabulary": {"interests": "interests@1.1.0", "aspects": "aspects@1.2.0"},
  "query": {"parent_asins": ["B0164RRP14", "NOT_A_REAL_ASIN"], "query_interests": ["fishing_hunting"], "max_reviews": 1},
  "warnings": [
    {"code": "unknown_interest", "value": "ponies", "message": "not a label in interests.json; ignored"},
    {"code": "unknown_asin", "value": "NOT_A_REAL_ASIN", "index": 1, "message": "parent_asin is not in the knowledge graph"}
  ],
  "results": ["... one object per input ASIN, same order ..."]
}
```

Warning codes: `unknown_asin`, `invalid_asin` (empty or non-string; `index` says which input), `unknown_interest`, `normalized_interest`, `max_reviews_capped`. Warnings are informational: a call with bad ASINs or labels still succeeds.

### One result

Every result has **all** of the keys below. Lists are empty and objects are `null` when there is no evidence, so you never need `KeyError` handling.

| Key | Meaning |
|---|---|
| `index`, `parent_asin`, `found` | Position in the input, the ASIN, and whether it is in the KG. |
| `product` | `title`, `domain` (`Books`, `Handmade Products` or `Toys & Games`), `price`, `average_rating`, `rating_number`, `store`, `age_min`, `age_max`, `review_count`, `average_review_rating`. Any of these values may be `null`. |
| `brand` | `{id, name, kind, provenance}` where `kind` is `author` for Books, otherwise `brand`. `null` when the listing has no real brand (e.g. "Generic"). |
| `category` | `{id, path, from_domain_fallback, provenance}`. `path` runs from the root to the leaf. `from_domain_fallback: true` means the listing had no category, so only the domain is known. |
| `interests` | Interests the product matches: `label`, `display_name`, `match_type`, `confidence`, `matches_query`, `evidence` (up to 3 provenance items), `path_id`. |
| `query_interest_match` | `{requested, matched, unmatched}`, or `null` when no valid `query_interests` were given. |
| `attributes` | Objective facts: `label`, `display_name`, `basis`, `confidence`, `evidence`, `path_id`. `basis` is `structured_metadata` (details/category/domain) or `seller_listing` (title or bullet text). |
| `review_qualities` | **Aggregated review conclusions.** See section 5. |
| `review_concerns` | **Aggregated negative conclusions.** See section 5. |
| `review_observations` | **Individual review statements.** Each has `review_id`, `aspect`, `polarity`, `rating`, `verified_purchase`, `helpful_votes`, `rating_conflict`, `snippets`, `confidence` and `path_id`. There are at most `max_reviews` per aspect. |
| `review_summary` | `reviews_in_graph`, `verified_reviews_in_graph`, `review_count_in_corpus` (all reviews on Amazon; the graph holds a sample of up to 5), `aspects_with_observations`, `observations_omitted` (per aspect, beyond `max_reviews`), and `aspect_status` (per aspect: `supported`, `concern`, `insufficient` or `neutral_only`). |
| `grounding` | `assertable_claims` (claims you may state) and `cautions` (things to mention carefully or avoid). Each item has `text` and `path_id`, and claims also have `basis`. |
| `reasoning_paths` | Traceable paths, one per claim, observation and concern: `{id, kind, claim, assertable, edges:[{src, rel, dst}]}`. Each `path_id` elsewhere refers to one of these. Edges use graph node IDs such as `product:<asin>`, `interest:<label>`, `aspect:<label>`, `review:<asin>:<hash>`. |

A **provenance item** is `{source, fields, text, confidence, rule_id, context?}`:
- `source` says how the evidence was found: `metadata`, `category_map`, `details`, `domain`, `title_lexicon`, `feature_lexicon` or `review_text`.
- `fields` names the corpus fields it came from, e.g. `["title"]` or `["features[2]"]`.
- `text` is the matched evidence, and `context` is the surrounding text when that differs.

An abridged real result from the sample (`max_reviews=1`):

```json
{
  "index": 0, "parent_asin": "B0164RRP14", "found": true,
  "product": {"title": "Personalized Fishing Gifts - Bass Fish Ornament - Fish Christmas Tree Ornaments",
              "domain": "Handmade Products", "price": 11.99, "average_rating": 4.7, "rating_number": 18,
              "store": "Palmetto Engraving", "age_min": null, "age_max": null,
              "review_count": 10, "average_review_rating": 5.0},
  "brand": {"id": "brand:palmetto-engraving", "name": "Palmetto Engraving", "kind": "brand",
            "provenance": {"source": "metadata", "fields": ["brand_or_author"], "text": "Palmetto Engraving", "confidence": 1.0, "rule_id": null}},
  "category": {"id": "category:handmade-products/home-and-kitchen/home-decor/decorative-accessories/ornaments",
               "path": ["Handmade Products", "Home & Kitchen", "Home Décor", "Decorative Accessories", "Ornaments"],
               "from_domain_fallback": false, "provenance": {"source": "metadata", "fields": ["categories"], "text": "Handmade Products > Home & Kitchen > ...", "confidence": 1.0, "rule_id": null}},
  "interests": [
    {"label": "fishing_hunting", "display_name": "Fishing & Hunting", "match_type": "themed", "confidence": 0.7, "matches_query": true,
     "evidence": [{"source": "title_lexicon", "fields": ["title"], "text": "Fishing", "confidence": 0.7, "rule_id": "fishing_hunting.title",
                   "context": "Personalized Fishing Gifts - Bass Fish Ornament - Fish Chris"}],
     "path_id": "p2"}
  ],
  "query_interest_match": {"requested": ["fishing_hunting"], "matched": ["fishing_hunting"], "unmatched": []},
  "attributes": [
    {"label": "material_wood", "display_name": "Wood", "basis": "seller_listing", "confidence": 0.6,
     "evidence": [{"source": "feature_lexicon", "fields": ["features[2]"], "text": "Birch Wood", "confidence": 0.6, "rule_id": "material_wood.features",
                   "context": "AMENT SPECIFICATIONS: Material: Natural Birch Wood; Thickness: 1/8 inch; ..."}],
     "path_id": "p5"}
  ],
  "review_qualities": [
    {"label": "craftsmanship", "display_name": "Craftsmanship", "conclusion": "supported", "polarity": "positive", "confidence": 0.7,
     "counts": {"positive": 2, "negative": 0, "mixed": 0, "neutral": 0, "rating_conflict": 0, "positive_verified": 2, "reviews_in_graph": 5},
     "supporting_reviews": [{"review_id": "review:B0164RRP14:99a280097b51992b", "text": "This is a very high quality product.", "rating": 5.0, "verified_purchase": true}],
     "supporting_reviews_omitted": 1, "path_id": "p8"}
  ],
  "review_concerns": [],
  "review_observations": [
    {"review_id": "review:B0164RRP14:3888145c3c0491d1", "aspect": "gift_reception", "aspect_display_name": "Gift reception",
     "polarity": "positive", "rating": 5.0, "verified_purchase": true, "helpful_votes": 0, "rating_conflict": false,
     "snippets": [{"text": "He loved it.", "polarity": "positive", "confidence": 0.7, "rule_id": "gift_reception.review.positive:direct"}],
     "confidence": 0.7, "path_id": "p7"}
  ],
  "review_summary": {"reviews_in_graph": 5, "verified_reviews_in_graph": 5, "review_count_in_corpus": 10,
                     "aspects_with_observations": ["craftsmanship", "gift_reception"],
                     "observations_omitted": {"craftsmanship": 1, "gift_reception": 1},
                     "aspect_status": {"craftsmanship": "supported", "gift_reception": "supported"}},
  "grounding": {
    "assertable_claims": [
      {"text": "Has a Fishing & Hunting theme (stated in the listing)", "basis": "interest_themed", "path_id": "p2"},
      {"text": "Handcrafted (product metadata)", "basis": "structured_metadata", "path_id": "p3"},
      {"text": "The listing says it is personalizable", "basis": "seller_listing", "path_id": "p4"},
      {"text": "2 independent reviews praise its craftsmanship", "basis": "review_aggregate", "path_id": "p8"}
    ],
    "cautions": []
  },
  "reasoning_paths": [
    {"id": "p2", "kind": "interest", "claim": "Has a Fishing & Hunting theme (stated in the listing)", "assertable": true,
     "edges": [{"src": "product:B0164RRP14", "rel": "MATCHES", "dst": "interest:fishing_hunting"}]},
    {"id": "p7", "kind": "review_observation", "claim": "One review gives positive evidence about Gift reception", "assertable": false,
     "edges": [{"src": "review:B0164RRP14:3888145c3c0491d1", "rel": "ABOUT", "dst": "product:B0164RRP14"},
               {"src": "review:B0164RRP14:3888145c3c0491d1", "rel": "SUPPORTS", "dst": "aspect:gift_reception"}]}
  ]
}
```

A product with a concern (`B09XDX9Q8T`, a wooden jigsaw):

```json
"review_concerns": [
  {"label": "ease_of_use", "display_name": "Ease of use", "status": "concern", "polarity": "negative",
   "counts": {"positive": 0, "negative": 2, "mixed": 0, "neutral": 0, "rating_conflict": 0},
   "examples": [{"review_id": "review:B09XDX9Q8T:46b498180047ec7a", "rating": 2.0, "verified_purchase": false,
                 "text": "... the little tabs and slots can be difficult to put together ... the puzzle is too frustrating to be fun."}],
   "path_id": "p10"}
],
"grounding": {
  "assertable_claims": [{"text": "Matches the interest Puzzles (product category)", "basis": "interest_primary", "path_id": "p2"},
                        {"text": "The listing says it is made of wood", "basis": "seller_listing", "path_id": "p3"}],
  "cautions": [{"text": "2 reviews report problems with ease of use", "path_id": "p10"}]
}
```

A missing product has `found: false`, `product`, `brand`, `category`, `query_interest_match` and `review_summary` set to `null`, every list empty, and a matching `unknown_asin` warning. **Never describe a product that came back with `found: false`.**

## 5. Interpreting the evidence

### Aggregated conclusions vs individual observations

| | Comes from | What it means | Can you assert it? |
|---|---|---|---|
| `review_qualities` | at least 2 different positive reviews, outnumbering negative + mixed reviews at least 2 to 1 | "Reviewers say it has good X" | **Yes**, with the count ("2 reviewers praise…"). |
| `review_concerns` | at least 2 negative reviews, more negatives than positives | "Several reviewers report problems with X" | Never as a positive. Mention it as a drawback, or rank the product lower. |
| `review_observations` | one review's statement | "One reviewer said …" | **Not as a general claim.** Quote it as one person's opinion if you use it at all. |

- `review_qualities` is **only ever positive** (`conclusion: "supported"`, `polarity: "positive"`). Its supporting reviews are clean positive statements: never mixed, and never praise inside a 1–2★ review (`rating_conflict`).
- `review_concerns` is **only ever negative**. A concern never appears as a quality for the same aspect.
- `review_observations` keep their own `polarity` (`positive`, `negative`, `neutral` or `mixed`). Read `mixed` as "this reviewer said both good and bad things"; it is never support. `rating_conflict: true` means the text sounds positive but the reviewer gave 1–2★, so don't use it as praise.
- **Star ratings alone are never evidence.** `average_rating` is listing metadata; quote it as such if you like.

### Which evidence is safe for grounded recommendations

1. **Safe to state:** everything in `grounding.assertable_claims` (all of them have `assertable: true` paths). Use the phrasing that matches `basis`:
   - `interest_primary`: the product is about or for the interest ("a fishing lure", "a history book").
   - `interest_themed`: the product carries the interest as a motif or license ("a dog-themed ring"). Say "themed", not "for fishing".
   - `structured_metadata`: a product fact.
   - `seller_listing`: say "the listing says…". It is the seller's statement, not a verified fact.
   - `review_aggregate`: "N reviewers praise its …".
2. **State carefully:** `grounding.cautions`. Review concerns, and qualities that some reviews dispute ("1 review(s) disagree about durability").
3. **Context only:** `review_observations`, `review_summary` and the `product` metadata.
4. **Never:** invent qualities, materials or interests that aren't in the response; turn a concern, mixed or negative observation into praise; or describe a `found: false` product.

`evidence_to_text(result)` in `knowledge_graph.api` produces exactly categories 1 and 2 plus short verbatim quotes, each tagged with its path ID (`[p8]`). That lets the LLM cite evidence and lets you verify every sentence against `reasoning_paths`.

### Confidence scores

All `confidence` values are **heuristic rule scores in [0, 1], not probabilities**, and no precision has been measured yet. They rank evidence strength:

| Score | Evidence |
|---|---|
| 1.0 | copied directly from metadata (brand, category) |
| 0.9–0.95 | category rules and structured `details` fields |
| 0.7–0.85 | listing title phrase (0.7 for interests, 0.8 for attributes); a direct statement in one review (0.7) |
| 0.6 | Books title phrase; listing bullet text; a negated positive in a review ("not sturdy") |
| 0.5 | feature text that only corroborates other evidence; a hedged or neutral review statement |

An edge's confidence is the **max** of its evidence items, never a sum. For review conclusions, the `counts` are a better strength signal than `confidence`. Use scores for ordering or thresholds you tune, not as probabilities shown to users.

### Optional and nullable fields

- Always present: every key listed in section 4.
- May be `null`: `product` / `brand` / `category` / `review_summary` / `query_interest_match` (see above), any `product` value, `rating`, `verified_purchase`, `helpful_votes`, `text` in provenance, and `rule_id` for metadata items.
- Only present when it adds information: `context` in a provenance item, and `normalized` / `index` / `message` in warnings.
- New keys may be added in a minor contract version (`kg-evidence/1.x`). Ignore keys you don't know. Removing or renaming a key requires `kg-evidence/2.0`.

## 6. Aligning interest extraction with `vocab/interests.json`

- **The labels are the contract.** Use `kg.interest_labels()` (or the `label` fields of `vocab/interests.json`) as an enum in the extraction schema, and the aliases and descriptions from `kg.interest_vocabulary()` as few-shot hints.
- **Record the version** `vocabulary.interests` with your outputs. A label change bumps the minor version (added labels) or major version (renamed or removed labels). Labels have been unchanged since 1.0.0.
- **Map, don't invent.** "Loves Star Wars" becomes `scifi_fantasy`, plus "Star Wars" in your free-text query. "Dog mom" becomes `dogs`. "Likes reading" is a domain preference (Books), not an interest.
- **If nothing fits,** send no interest rather than the closest guess. Unknown labels are dropped anyway, and you'll see an `unknown_interest` warning.
- **Need a new label?** Ask the KG owner. It needs vocabulary rules and a version bump, not just a new string.

## 7. Limitations of the current inference

- **Rule-based.** Interests and attributes come from category, details, title and bullet rules. Review aspects come from phrase cues with negation, hedge and contrast handling. A trained classifier can replace the review rules without changing this contract.
- **Unmeasured precision.** Labeling sheets are in `eval/`.
- **Review coverage is small in the sample.** At most 5 reviews per product, so only about 9% of products have a `review_qualities` entry and about 1% have a concern. "No review conclusion" means *unknown*, not "bad".
- **Seller-stated attributes are unverified.** Materials of minor components can still slip through as listing attributes.
- **Possible review misattributions.** The review rules may attribute a comment about the box, the seller or another product to this product, or miss sarcasm. Individual observations should therefore not be generalized.
- **Interest coverage:** about 52% of Books, 44% of Toys and 14% of Handmade products (much Handmade jewelry and decor has no hobby signal). Party supplies and nursery decor intentionally get no text-based interests.
- **Not in the KG:** recipient, occasion, budget and age reasoning. Age bounds come straight from metadata and are often `null`.
- **Graph size:** the in-memory graph suits the sample and moderate exports. The full 3.78M-product corpus **has not been built yet**; the plan is to build it once on Sol with `--out` and load it with `export_dir`. Memory use at that scale is untested. A database-backed store would implement the same `KGService` methods.

## 8. Contract checklist for the LLM component

- [ ] Load once at startup: `kg = load_kg(...)`.
- [ ] Send `parent_asins` in ranking order. Expect results in the same order.
- [ ] Send interest **labels** only, and log the `warnings`.
- [ ] Ground product sentences in `grounding.assertable_claims`, and handle `grounding.cautions` honestly.
- [ ] Skip or flag `found: false` results.
- [ ] Never treat `confidence` as a probability, or a concern or observation as praise.
