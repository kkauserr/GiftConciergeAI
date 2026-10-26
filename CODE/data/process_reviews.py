import json
import sys
from collections import defaultdict

if len(sys.argv) != 3:
    print("Usage: python process_reviews.py <input_reviews> <output_file>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

# Store aggregated review information by product
products = defaultdict(lambda: {
    "review_count": 0,
    "verified_review_count": 0,
    "helpful_votes": 0,
    "rating_sum": 0.0,
    "review_samples": []
})

MAX_REVIEW_SAMPLES = 5

total_reviews = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        try:
            review = json.loads(line)
        except json.JSONDecodeError:
            continue

        parent_asin = review.get("parent_asin")

        if not parent_asin:
            continue

        total_reviews += 1

        product = products[parent_asin]

        product["review_count"] += 1

        rating = review.get("rating")
        if rating is not None:
            product["rating_sum"] += float(rating)

        if review.get("verified_purchase") is True:
            product["verified_review_count"] += 1

        helpful_vote = review.get("helpful_vote", 0)

        if isinstance(helpful_vote, (int, float)):
            product["helpful_votes"] += helpful_vote

        # Keep only a small number of useful review texts
        text = review.get("text", "")

        if (
            isinstance(text, str)
            and text.strip()
            and len(product["review_samples"]) < MAX_REVIEW_SAMPLES
        ):
            product["review_samples"].append(text.strip())


with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for parent_asin, data in products.items():

        review_count = data["review_count"]

        average_review_rating = (
            data["rating_sum"] / review_count
            if review_count > 0
            else None
        )

        record = {
            "parent_asin": parent_asin,
            "review_count": review_count,
            "average_review_rating": round(average_review_rating, 2),
            "verified_review_count": data["verified_review_count"],
            "helpful_votes": data["helpful_votes"],
            "review_samples": data["review_samples"]
        }

        output.write(json.dumps(record, ensure_ascii=False) + "\n")


print("\n===== REVIEW PROCESSING COMPLETE =====")
print(f"Reviews processed: {total_reviews:,}")
print(f"Unique products with reviews: {len(products):,}")
print(f"Output file: {OUTPUT_FILE}")
