import pandas as pd

PATH = "EVALUATIONS/queries/precision_labels.csv"

df = pd.read_csv(PATH)

df["grade"] = pd.to_numeric(df["grade"], errors="coerce")

if df["grade"].isna().any():
    print("ERROR: Some grades are missing or invalid.")
    print(df[df["grade"].isna()][["qid", "title", "grade"]])
    raise SystemExit(1)

def precision_for_system(system_col):
    scores = []

    for qid, group in df.groupby("qid"):
        ranked = group[group[system_col].notna()].copy()

        ranked[system_col] = pd.to_numeric(
            ranked[system_col], errors="coerce"
        )

        ranked = ranked.sort_values(system_col).head(5)

        relevant = (ranked["grade"] == 1).sum()
        precision = relevant / 5

        scores.append((qid, precision, relevant))

    return scores

bm25 = precision_for_system("bm25")
full = precision_for_system("full")

print("\n--- Precision@5 Evaluation ---")

print("\nBM25:")
for qid, p, relevant in bm25:
    print(f"{qid}: {relevant}/5 = {p:.3f}")

print("\nFull Hybrid:")
for qid, p, relevant in full:
    print(f"{qid}: {relevant}/5 = {p:.3f}")

bm25_mean = sum(x[1] for x in bm25) / len(bm25)
full_mean = sum(x[1] for x in full) / len(full)

print("\n--- Final ---")
print(f"BM25 Precision@5:        {bm25_mean:.3f}")
print(f"Full Hybrid Precision@5: {full_mean:.3f}")
print(f"Difference:              {full_mean - bm25_mean:+.3f}")