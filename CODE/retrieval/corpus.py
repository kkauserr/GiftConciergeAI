"""Load product records by parent_asin (for ranking, explanation and the demo)."""

import json
from pathlib import Path


def iter_products(path):
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_products(path, asins=None):
    """Return {parent_asin: record}.

    Pass `asins` to keep only those products. That is how to pull the top-K
    records out of the full 27 GB corpus without holding all of it in memory
    (it is still a full scan, so prefer a keyed store on Sol for repeated lookups).
    """
    wanted = set(asins) if asins is not None else None
    products = {}
    for record in iter_products(Path(path)):
        asin = record.get("parent_asin")
        if asin and (wanted is None or asin in wanted):
            products[asin] = record
    return products
