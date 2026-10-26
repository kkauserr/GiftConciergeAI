import json
import sys


def has_value(value):
    """Return True if a metadata field contains useful information."""
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return True


if len(sys.argv) != 2:
    print("Usage: python analyze_metadata.py <metadata_file>")
    sys.exit(1)


file_path = sys.argv[1]

fields = [
    "title",
    "price",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "categories",
    "details",
    "bought_together"
]

counts = {field: 0 for field in fields}

total = 0
usable = 0
bad_records = 0


with open(file_path, "r", encoding="utf-8") as f:

    for line in f:

        try:
            product = json.loads(line)
        except json.JSONDecodeError:
            bad_records += 1
            continue

        total += 1

        # Count field availability
        for field in fields:
            if has_value(product.get(field)):
                counts[field] += 1

        # Simple Gift Concierge feasibility rule
        #
        # Product must have:
        # title
        # price
        # average rating
        # AND either features or description

        if (
            has_value(product.get("title"))
            and has_value(product.get("price"))
            and has_value(product.get("average_rating"))
            and (
                has_value(product.get("features"))
                or has_value(product.get("description"))
            )
        ):
            usable += 1


print("\n===== METADATA SUMMARY =====")
print(f"Total products: {total:,}")
print(f"Malformed records: {bad_records:,}")

print("\nField availability:")

for field in fields:
    percentage = (counts[field] / total * 100) if total else 0
    print(f"{field:20} {counts[field]:10,}  ({percentage:6.2f}%)")


usable_percentage = (usable / total * 100) if total else 0

print("\nGift Concierge feasibility:")
print(f"Usable products: {usable:,} ({usable_percentage:.2f}%)")
