import json
import os

# ---------------------------------------------------------
# Input files
# ---------------------------------------------------------

DATA_DIR = os.environ.get(
    "GIFT_DATA_DIR",
    os.path.join(
        os.environ.get("SCRATCH", os.getcwd()),
        "gift-concierge",
        "data"
    )
)

INPUT_FILES = {
    "Books": os.path.join(DATA_DIR, "meta_Books.jsonl"),
    "Handmade Products": os.path.join(
        DATA_DIR, "meta_Handmade_Products.jsonl"
    ),
    "Toys & Games": os.path.join(
        DATA_DIR, "meta_Toys_and_Games.jsonl"
    ),
}

OUTPUT_FILE = os.path.join(DATA_DIR, "processed_products.jsonl")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def has_value(value):
    """Return True when a field contains useful information."""
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return True


def text_from_field(value):
    """Convert different Amazon metadata fields into searchable text."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)

    if isinstance(value, dict):
        return " ".join(
            f"{key} {val}"
            for key, val in value.items()
            if val
        )

    return str(value)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

total_seen = 0
total_kept = 0

category_counts = {}

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for domain, file_path in INPUT_FILES.items():

        print(f"\nProcessing {domain}...")

        seen = 0
        kept = 0

        with open(file_path, "r", encoding="utf-8") as input_file:

            for line in input_file:

                try:
                    product = json.loads(line)
                except json.JSONDecodeError:
                    continue

                seen += 1
                total_seen += 1

                # -----------------------------------------
                # Apply our preliminary usability rules
                # -----------------------------------------

                if not has_value(product.get("parent_asin")):
                    continue

                if not has_value(product.get("title")):
                    continue

                if not has_value(product.get("price")):
                    continue

                if not has_value(product.get("average_rating")):
                    continue

                if not (
                    has_value(product.get("features"))
                    or has_value(product.get("description"))
                ):
                    continue

                # -----------------------------------------
                # Build searchable product text
                # -----------------------------------------

                text_parts = [
                    text_from_field(product.get("title")),
                    text_from_field(product.get("features")),
                    text_from_field(product.get("description")),
                    text_from_field(product.get("categories")),
                    text_from_field(product.get("details")),
                ]

                product_text = " ".join(
                    part for part in text_parts if part
                )

                # -----------------------------------------
                # Standardized product record
                # -----------------------------------------

                cleaned_product = {
                    "parent_asin": product.get("parent_asin"),
                    "domain": domain,
                    "title": product.get("title"),
                    "price": product.get("price"),
                    "average_rating": product.get("average_rating"),
                    "rating_number": product.get("rating_number"),
                    "features": product.get("features"),
                    "description": product.get("description"),
                    "categories": product.get("categories"),
                    "details": product.get("details"),
                    "product_text": product_text,
                }

                output.write(
                    json.dumps(cleaned_product, ensure_ascii=False) + "\n"
                )

                kept += 1
                total_kept += 1

        category_counts[domain] = {
            "seen": seen,
            "kept": kept,
        }

        print(f"  Products examined: {seen:,}")
        print(f"  Products retained: {kept:,}")


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n===== PREPROCESSING COMPLETE =====")

for domain, counts in category_counts.items():

    percentage = (
        counts["kept"] / counts["seen"] * 100
        if counts["seen"]
        else 0
    )

    print(
        f"{domain:20} "
        f"{counts['kept']:10,} / "
        f"{counts['seen']:10,} "
        f"({percentage:.2f}%)"
    )

print(f"\nTotal products examined: {total_seen:,}")
print(f"Total products retained: {total_kept:,}")
print(f"Output file: {OUTPUT_FILE}")
