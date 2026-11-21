import ast
import json
import re

INPUT_FILE = "/scratch/jkauser/gift-concierge/data/final_product_corpus_v2.jsonl"
OUTPUT_FILE = "/scratch/jkauser/gift-concierge/data/final_product_corpus_v2_fixed.jsonl"

# Load only parse_age() from preprocess_products.py
with open("CODE/data/preprocess_products.py") as f:
    source = f.read()

tree = ast.parse(source)

func = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "parse_age"
)

module = ast.Module(body=[func], type_ignores=[])
namespace = {"re": re}

exec(
    compile(module, "CODE/data/preprocess_products.py", "exec"),
    namespace
)

parse_age = namespace["parse_age"]

processed = 0
changed = 0
still_invalid = 0

with open(INPUT_FILE) as src, open(OUTPUT_FILE, "w") as dst:
    for line in src:
        row = json.loads(line)

        old_min = row.get("age_min")
        old_max = row.get("age_max")

        new_min, new_max = parse_age(row.get("details", {}))

        row["age_min"] = new_min
        row["age_max"] = new_max

        if old_min != new_min or old_max != new_max:
            changed += 1

        if new_min is not None and new_max is not None and new_min > new_max:
            still_invalid += 1

        dst.write(json.dumps(row, ensure_ascii=False) + "\n")

        processed += 1

        if processed % 500000 == 0:
            print(f"Processed {processed:,} products...")

print("\n===== AGE REPAIR COMPLETE =====")
print(f"Products processed: {processed:,}")
print(f"Products with changed age values: {changed:,}")
print(f"Products still having age_min > age_max: {still_invalid:,}")
print(f"Output: {OUTPUT_FILE}")
