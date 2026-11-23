"""Build {parent_asin: main image URL} for a corpus from the raw Amazon metadata.

The processed corpus has no image URLs, so this scans the original
meta_*.jsonl files (on Sol) for the products in a corpus and keeps each one's
MAIN image. The backend attaches these URLs to recommendation cards.

    # 60K demo subset (defaults)
    python CODE/data/build_image_lookup.py

    # 900-product GitHub sample (commit the small output so graders get images)
    python CODE/data/build_image_lookup.py \
        --subset DATA/samples/sample_corpus_v2.jsonl \
        --out DATA/samples/sample_image_lookup.json
"""

import argparse
import json
import os

DATA_DIR = "/scratch/jkauser/gift-concierge/data"
SUBSET_PATH = f"{DATA_DIR}/dev_subset_v2.jsonl"
OUTPUT_PATH = f"{DATA_DIR}/dev60k_image_lookup.json"

METADATA_FILES = {
    "Books": "meta_Books.jsonl",
    "Handmade Products": "meta_Handmade_Products.jsonl",
    "Toys & Games": "meta_Toys_and_Games.jsonl",
}


def get_main_image(images):
    for image in images or []:
        if image.get("variant") == "MAIN":
            return (
                image.get("hi_res")
                or image.get("large")
                or image.get("thumb")
            )

    return None


def build(subset_path, output_path, meta_dir):
    # Collect the ASINs we actually need
    needed = {}

    with open(subset_path, "r") as f:
        for line in f:
            product = json.loads(line)

            asin = product.get("parent_asin")
            domain = product.get("domain")

            if asin and domain:
                needed[asin] = domain

    print(f"Need images for {len(needed):,} products")

    # Search original metadata
    image_lookup = {}

    for domain, filename in METADATA_FILES.items():

        domain_asins = {
            asin for asin, product_domain in needed.items()
            if product_domain == domain
        }

        print(f"\nScanning {domain} for {len(domain_asins):,} products...")

        found = 0

        with open(os.path.join(meta_dir, filename), "r") as f:
            for line in f:
                product = json.loads(line)

                asin = product.get("parent_asin")

                if asin not in domain_asins:
                    continue

                image_url = get_main_image(product.get("images"))

                if image_url:
                    image_lookup[asin] = image_url
                    found += 1

        print(f"Found images: {found:,}")

    with open(output_path, "w") as f:
        json.dump(image_lookup, f)

    print(f"\nTotal images: {len(image_lookup):,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default=SUBSET_PATH, help="corpus JSONL whose products need images")
    parser.add_argument("--out", default=OUTPUT_PATH, help="output JSON {parent_asin: image_url}")
    parser.add_argument("--meta-dir", default=DATA_DIR, help="folder with the raw meta_*.jsonl files")
    args = parser.parse_args()
    build(args.subset, args.out, args.meta_dir)
