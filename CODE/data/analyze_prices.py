import json
import os
import re
from collections import Counter

DATA_DIR = os.environ.get(
    "GIFT_DATA_DIR",
    os.path.join(
        os.environ.get("SCRATCH", os.getcwd()),
        "gift-concierge",
        "data"
    )
)

FILES = {
    "Books": "meta_Books.jsonl",
    "Handmade Products": "meta_Handmade_Products.jsonl",
    "Toys & Games": "meta_Toys_and_Games.jsonl",
}


def classify_price(value):
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "other"

    if isinstance(value, (int, float)):
        if value <= 0:
            return "numeric_zero_or_negative"
        if value > 500:
            return "numeric_over_500"
        return "numeric_valid"

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return "empty_string"

        if value.lower() in {"none", "null", "nan", "n/a"}:
            return "null_like_string"

        nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))

        if len(nums) == 0:
            return "string_no_number"

        if len(nums) > 1:
            return "string_multiple_numbers"

        try:
            number = float(nums[0])

            if number <= 0:
                return "string_zero_or_negative"

            if number > 500:
                return "string_over_500"

            return "string_single_number"

        except ValueError:
            return "string_parse_error"

    return "other"


for domain, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)

    print(f"\n{'=' * 60}")
    print(domain)
    print(f"{'=' * 60}")

    counts = Counter()
    examples = {}
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue

            total += 1

            price = record.get("price")
            category = classify_price(price)

            counts[category] += 1

            if category not in examples and price is not None:
                examples[category] = price

    print(f"Total products: {total:,}")

    for category, count in counts.most_common():
        percent = (count / total * 100) if total else 0

        print(
            f"{category:30} "
            f"{count:12,} "
            f"{percent:7.2f}%"
        )

        if category in examples:
            print(f"    Example: {repr(examples[category])}")
