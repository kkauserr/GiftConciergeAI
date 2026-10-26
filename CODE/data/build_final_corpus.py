import json
import os

DATA_DIR = os.environ.get(
    "GIFT_DATA_DIR",
    os.path.join(
        os.environ.get("SCRATCH", os.getcwd()),
        "gift-concierge",
        "data"
    )
)

PRODUCT_FILE = os.path.join(DATA_DIR, "processed_products.jsonl")

REVIEW_FILES = [
    os.path.join(DATA_DIR, "Books_review_summary.jsonl"),
    os.path.join(DATA_DIR, "Handmade_review_summary.jsonl"),
    os.path.join(DATA_DIR, "Toys_review_summary.jsonl"),
]

OUTPUT_FILE = os.path.join(DATA_DIR, "final_product_corpus.jsonl")


print("Loading review summaries...")

reviews = {}

for review_file in REVIEW_FILES:

    print(f"  Loading {os.path.basename(review_file)}")

    with open(review_file, "r", encoding="utf-8") as f:

        for line in f:

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            parent_asin = review.get("parent_asin")

            if parent_asin:
                reviews[parent_asin] = review


print(f"\nReview summaries loaded: {len(reviews):,}")
print("\nBuilding final corpus...")


products_seen = 0
products_written = 0
products_with_reviews = 0


with open(PRODUCT_FILE, "r", encoding="utf-8") as products, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for line in products:

        try:
            product = json.loads(line)
        except json.JSONDecodeError:
            continue

        products_seen += 1

        parent_asin = product.get("parent_asin")

        review = reviews.get(parent_asin)

        if review:
            products_with_reviews += 1

        final_record = {
            "parent_asin": parent_asin,
            "domain": product.get("domain"),
            "title": product.get("title"),
            "price": product.get("price"),
            "average_rating": product.get("average_rating"),
            "rating_number": product.get("rating_number"),

            # Main text used by BM25 / embeddings
            "product_text": product.get("product_text"),

            # Aggregated review evidence
            "review_count": review.get("review_count", 0) if review else 0,
            "average_review_rating": (
                review.get("average_review_rating")
                if review else None
            ),
            "verified_review_count": (
                review.get("verified_review_count", 0)
                if review else 0
            ),
            "helpful_votes": (
                review.get("helpful_votes", 0)
                if review else 0
            ),
            "review_samples": (
                review.get("review_samples", [])
                if review else []
            ),
        }

        output.write(
            json.dumps(final_record, ensure_ascii=False) + "\n"
        )

        products_written += 1


print("\n===== FINAL CORPUS COMPLETE =====")
print(f"Products examined: {products_seen:,}")
print(f"Products written: {products_written:,}")
print(f"Products with review evidence: {products_with_reviews:,}")

coverage = (
    products_with_reviews / products_written * 100
    if products_written
    else 0
)

print(f"Review coverage: {coverage:.2f}%")
print(f"Output file: {OUTPUT_FILE}")
