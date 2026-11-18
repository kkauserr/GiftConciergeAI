import json
import sys
import html
from collections import defaultdict

def clean_review_text(text):
    if not isinstance(text, str):
        return ""

    # Convert HTML entities such as &amp; to &
    text = html.unescape(text)

    # Replace common HTML line breaks with spaces
    text = text.replace("<br />", " ")
    text = text.replace("<br/>", " ")
    text = text.replace("<br>", " ")

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()

if len(sys.argv) != 5:
    print(
        "Usage: python process_reviews.py "
        "<input_reviews> <processed_products> <domain> <output_file>"
    )
    sys.exit(1)

INPUT_FILE = sys.argv[1]
PROCESSED_PRODUCTS_FILE = sys.argv[2]
DOMAIN = sys.argv[3]
OUTPUT_FILE = sys.argv[4]

retained_ids = set()

print(f"\nLoading retained product IDs for {DOMAIN}...")

with open(PROCESSED_PRODUCTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            product = json.loads(line)
        except json.JSONDecodeError:
            continue

        if product.get("domain") == DOMAIN:
            parent_asin = product.get("parent_asin")

            if parent_asin:
                retained_ids.add(parent_asin)

print(f"Retained product IDs loaded: {len(retained_ids):,}")



# Store aggregated review information by product
products = defaultdict(lambda: {
    "review_count": 0,
    "rated_review_count": 0,
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
    # Ignore reviews for products removed during metadata preprocessing.
        if parent_asin not in retained_ids:
            continue

        product = products[parent_asin]

        product["review_count"] += 1

        rating = review.get("rating")
        if rating is not None:
            product["rating_sum"] += float(rating)
            product["rated_review_count"] +=1 

        if review.get("verified_purchase") is True:
            product["verified_review_count"] += 1

        helpful_vote = review.get("helpful_vote", 0)

        if isinstance(helpful_vote, (int, float)):
            product["helpful_votes"] += helpful_vote

        # Keep up to 5 most recent non-empty reviews
        text = clean_review_text(review.get("text", ""))

        if isinstance(text, str) and text.strip():

            review_sample = {
                "title": review.get("title"),
                "text": text.strip(),
                "rating": review.get("rating"),
                "helpful_vote": review.get("helpful_vote", 0),
                "verified_purchase": review.get("verified_purchase"),
                "timestamp": review.get("timestamp"),
            }

            product["review_samples"].append(review_sample)

            # Sort reviews from newest to oldest.
            product["review_samples"].sort(
                key=lambda r: (
                    r["timestamp"]
                    if isinstance(r["timestamp"], (int, float))
                    else 0
                ),
                reverse=True,
            )

            # Keep only the 5 most recent reviews.
            if len(product["review_samples"]) > MAX_REVIEW_SAMPLES:
                product["review_samples"].pop()


with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for parent_asin, data in products.items():

        review_count = data["review_count"]
        rated_review_count = data["rated_review_count"]
        average_review_rating = (
            data["rating_sum"] /rated_review_count
            if rated_review_count > 0
            else None
        )

        record = {
            "parent_asin": parent_asin,
            "review_count": review_count,
            "rated_review_count": rated_review_count,
            "average_review_rating": (
                round(average_review_rating, 2)
                if average_review_rating is not None
                else None
            ),
            "verified_review_count": data["verified_review_count"],
            "helpful_votes": data["helpful_votes"],
            "review_samples": data["review_samples"]
        }

        output.write(json.dumps(record, ensure_ascii=False) + "\n")


print("\n===== REVIEW PROCESSING COMPLETE =====")
print(f"Reviews processed: {total_reviews:,}")
print(f"Unique products with reviews: {len(products):,}")
print(f"Output file: {OUTPUT_FILE}")
