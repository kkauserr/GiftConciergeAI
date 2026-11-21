import argparse
import json
import random

SLIM_DROP = ["description", "details", "features", "review_samples"]


def main():
    p = argparse.ArgumentParser(description="Random per-domain subset of the final corpus for ranking development")
    p.add_argument("--input", default="/scratch/jkauser/gift-concierge/data/final_product_corpus_v2_fixed.jsonl")
    p.add_argument("--output", default="dev_subset_v2.jsonl")
    p.add_argument("--per-domain", type=int, default=20000)
    p.add_argument("--seed", type=int, default=573)
    p.add_argument("--full-records", action="store_true", help="keep description/details/features/reviews")
    args = p.parse_args()

    rng = random.Random(args.seed)
    reservoirs, seen = {}, {}

    with open(args.input, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                domain = json.loads(line).get("domain")
            except json.JSONDecodeError:
                continue
            seen[domain] = seen.get(domain, 0) + 1
            res = reservoirs.setdefault(domain, [])
            if len(res) < args.per_domain:
                res.append(line)
            else:
                j = rng.randrange(seen[domain])
                if j < args.per_domain:
                    res[j] = line
            if n % 500000 == 0:
                print(f"Scanned {n:,} lines")

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for domain, lines in reservoirs.items():
            for line in lines:
                row = json.loads(line)
                if not args.full_records:
                    for k in SLIM_DROP:
                        row.pop(k, None)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"{domain}: {len(lines):,} sampled from {seen[domain]:,}")
    print(f"Wrote {written:,} products to {args.output}")


if __name__ == "__main__":
    main()
