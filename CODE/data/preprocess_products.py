import json
import os
import sys
import re
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
# Optional command-line overrides
# Usage:
# python preprocess_products.py [DATA_DIR] [OUTPUT_FILE]
# ---------------------------------------------------------

if len(sys.argv) >= 2:
    DATA_DIR = sys.argv[1]

    INPUT_FILES = {
        "Books": os.path.join(DATA_DIR, "meta_Books.jsonl"),
        "Handmade Products": os.path.join(
            DATA_DIR, "meta_Handmade_Products.jsonl"
        ),
        "Toys & Games": os.path.join(
            DATA_DIR, "meta_Toys_and_Games.jsonl"
        ),
    }

if len(sys.argv) >= 3:
    OUTPUT_FILE = sys.argv[2]

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def parse_price(value):
    """
    Convert an Amazon price into a positive numeric value.

    Examples:
        12.99        -> 12.99
        "from 3.40"  -> 3.40
        "—"          -> None
        0.0          -> None
        None         -> None
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    if isinstance(value, str):
        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            value.replace(",", "")
        )

        # Only accept strings containing one numeric price.
        if len(numbers) == 1:
            price = float(numbers[0])
            return price if price > 0 else None

    return None

def parse_age(details):
    """
    Extract minimum and maximum recommended ages in years.

    Returns:
        (age_min, age_max)

    Examples:
        "7 - 10 years"    -> (7.0, 10.0)
        "13 years and up" -> (13.0, None)
        204 months        -> (17.0, None)
    """
    if not isinstance(details, dict):
        return None, None

    # Most useful direct age fields, in priority order.
    age_keys = [
        "Reading age",
        "Manufacturer recommended age",
        "Our Recommended age",
        "Reading Interest (Min. Age)",
    ]

    for key in age_keys:
        value = details.get(key)

        if not value:
            continue

        text = str(value).lower().replace(",", "")
        numbers = re.findall(r"\d+(?:\.\d+)?", text)

        if not numbers:
            continue

        numbers = [float(n) for n in numbers]

        # Example: "7 - 10 years"
        if len(numbers) >= 2:
            return numbers[0], numbers[1]

        # Example: "13 years and up"
        if len(numbers) == 1:
            return numbers[0], None

    # Some Amazon records store minimum age directly in months.
    months = details.get("Manufacturer Minimum Age (MONTHS)")

    if months:
        try:
            age_min = float(months) / 12.0
            return round(age_min, 1), None
        except (TypeError, ValueError):
            pass

    return None, None

def filter_details(details):
    if not isinstance(details, dict):
        return {}

    # Fields that are mostly identifiers, logistics, or retrieval noise.
    excluded_keywords = [
        "asin",
        "isbn",
        "rank",
        "dimensions",
        "weight",
        "date",
        "model number",
        "item model",
        "product dimensions",
        "package dimensions",
    ]

    filtered = {}

    for key, value in details.items():
        key_lower = str(key).lower()

        if any(keyword in key_lower for keyword in excluded_keywords):
            continue

        if value is None or value == "":
            continue

        filtered[key] = value

    return filtered

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

drop_counts = {
    "missing_parent_asin": 0,
    "missing_title": 0,
    "invalid_price": 0,
    "missing_rating": 0,
    "missing_content": 0,
}

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
                    drop_counts["missing_parent_asin"] += 1
                    continue

                if not has_value(product.get("title")):
                    drop_counts["missing_title"] += 1
                    continue

                price_num = parse_price(product.get("price"))

                if price_num is None:
                    drop_counts["invalid_price"] += 1
                    continue

                if not has_value(product.get("average_rating")):
                    drop_counts["missing_rating"] += 1
                    continue

                if not (
                    has_value(product.get("features"))
                    or has_value(product.get("description"))
                ):
                    drop_counts["missing_content"] += 1
                    continue

                                # -----------------------------------------
                # Extract structured metadata
                # -----------------------------------------

                details = product.get("details") or {}
                filtered_details = filter_details(details)

                categories = product.get("categories") or []
                main_category = product.get("main_category")

                # Most specific category when available.
                # Fall back to main_category.
                specific_category = (
                    categories[-1]
                    if isinstance(categories, list) and categories
                    else main_category
                )

                # Books have a structured author object.
                # Other domains primarily use store/brand.
                author = product.get("author")
                brand_or_author = None

                if domain == "Books" and isinstance(author, dict):
                    brand_or_author = author.get("name")

                if not brand_or_author:
                    brand_or_author = product.get("store")

                if not brand_or_author and isinstance(details, dict):
                    brand_or_author = details.get("Brand")

                # Extract recommended age information.
                age_min, age_max = parse_age(details)

                # -----------------------------------------
                # Build retrieval text
                # -----------------------------------------

                # Full text for BM25 / lexical retrieval.
                bm25_parts = [
                    text_from_field(product.get("title")),
                    text_from_field(product.get("features")),
                    text_from_field(product.get("description")),
                    text_from_field(categories),
                    text_from_field(filtered_details),
                ]

                bm25_text = " ".join(
                    part for part in bm25_parts if part
                )

                # Shorter, cleaner text for semantic embeddings.
                features = product.get("features") or []

                if isinstance(features, list):
                    short_features = features[:3]
                else:
                    short_features = features

                age_text = ""

                if age_min is not None and age_max is not None:
                    age_text = f"Recommended age {age_min} to {age_max} years"
                elif age_min is not None:
                    age_text = f"Recommended age {age_min}+ years"

                embed_parts = [
                    text_from_field(product.get("title")),
                    text_from_field(specific_category),
                    text_from_field(categories),
                    text_from_field(brand_or_author),
                    age_text,
                    text_from_field(short_features),
                ]

                embed_text = " ".join(
                    part for part in embed_parts if part
                ) 

                # -----------------------------------------
                # Standardized product record
                # -----------------------------------------

                cleaned_product = {
                    "parent_asin": product.get("parent_asin"),
                    "domain": domain,
                    "title": product.get("title"),

                    # Price
                    "price": product.get("price"),
                    "price_num": price_num,

                    # Ratings
                    "average_rating": product.get("average_rating"),
                    "rating_number": product.get("rating_number"),

                    # Brand / author
                    "store": product.get("store"),
                    "brand_or_author": brand_or_author,

                    # Categories
                    "main_category": main_category,
                    "specific_category": specific_category,
                    "categories": categories,

                    # Age suitability
                    "age_min": age_min,
                    "age_max": age_max,

                    # Product content
                    "features": product.get("features"),
                    "description": product.get("description"),
                    "details": filtered_details,

                    # Retrieval text
                    "bm25_text": bm25_text,
                    "embed_text": embed_text,
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
print("\n===== DROP REASONS =====")

for reason, count in drop_counts.items():
    percentage = (
        count / total_seen * 100
        if total_seen
        else 0
    )

    print(f"{reason:25} {count:10,} ({percentage:.2f}%)")
