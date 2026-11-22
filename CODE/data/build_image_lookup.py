import json

SUBSET_PATH = "/scratch/jkauser/gift-concierge/data/dev_subset_v2.jsonl"
OUTPUT_PATH = "/scratch/jkauser/gift-concierge/data/dev60k_image_lookup.json"

METADATA_FILES = {
    "Books": "/scratch/jkauser/gift-concierge/data/meta_Books.jsonl",
    "Handmade Products": "/scratch/jkauser/gift-concierge/data/meta_Handmade_Products.jsonl",
    "Toys & Games": "/scratch/jkauser/gift-concierge/data/meta_Toys_and_Games.jsonl",
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


# Collect the ASINs we actually need
needed = {}

with open(SUBSET_PATH, "r") as f:
    for line in f:
        product = json.loads(line)

        asin = product.get("parent_asin")
        domain = product.get("domain")

        if asin and domain:
            needed[asin] = domain

print(f"Need images for {len(needed):,} products")


# Search original metadata
image_lookup = {}

for domain, metadata_path in METADATA_FILES.items():

    domain_asins = {
        asin for asin, product_domain in needed.items()
        if product_domain == domain
    }

    print(f"\nScanning {domain} for {len(domain_asins):,} products...")

    found = 0

    with open(metadata_path, "r") as f:
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


with open(OUTPUT_PATH, "w") as f:
    json.dump(image_lookup, f)

print(f"\nTotal images: {len(image_lookup):,}")
print(f"Saved to: {OUTPUT_PATH}")
