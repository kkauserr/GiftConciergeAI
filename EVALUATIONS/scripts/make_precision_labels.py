import csv
import json

RUNS = "EVALUATIONS/results/precision_temp/ablation_runs.json"
QUERIES = "EVALUATIONS/queries/precision_queries.json"
OUT = "EVALUATIONS/queries/precision_labels.csv"

runs = json.load(open(RUNS))
queries = {q["qid"]: q for q in json.load(open(QUERIES))}

rows = []

for qid, q in queries.items():
    seen = {}

    for system in ["bm25_only", "full"]:
        for r in runs[system][qid]:
            asin = r["parent_asin"]

            if asin not in seen:
                seen[asin] = {
                    "qid": qid,
                    "request": q["request"],
                    "parent_asin": asin,
                    "title": r["title"],
                    "domain": r["domain"],
                    "price": r["price"],
                    "bm25": "",
                    "full": "",
                    "grade": ""
                }

            seen[asin][system.replace("_only", "")] = r["rank"]

    rows.extend(seen.values())

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "qid", "request", "parent_asin", "title",
            "domain", "price", "bm25", "full", "grade"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
print("Grade each row: 1 = relevant, 0 = not relevant")