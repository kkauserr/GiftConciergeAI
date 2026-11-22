"""Normalize exclusions so the ranking filter can enforce them.

CODE/ranking/filters.py (owner: Disha) matches each exclusion as a whole word or
phrase against title + categories, and expands a few broad group names itself.

  * FILTER_GROUPS: expanded by filters.py. We only send the group name, so there
    is one word list and no double expansion ("things to wear" -> "wearable").
  * EXPANDED_GROUPS: filters.py doesn't know these, so we send concrete words,
    as its README asks.

Everything else passes through unchanged.
"""

FILTER_GROUPS = ("wearable", "clothing", "jewelry", "food", "decor")

EXPANDED_GROUPS = {
    "stuffed animals": ["stuffed animal", "stuffed", "plush", "plushie", "soft toy"],
    "scented": ["scented", "candle", "soap", "fragrance", "perfume", "incense",
                "essential oil", "bath bomb"],
    "electronic": ["electronic", "electric", "battery", "batteries", "remote control", "rc",
                   "rechargeable", "usb"],
}

EXCLUSION_GROUPS = FILTER_GROUPS + tuple(EXPANDED_GROUPS)

# Other ways users (or the LLM) phrase a group.
EXCLUSION_ALIASES = {
    "wearables": "wearable", "wearable items": "wearable", "wearable things": "wearable",
    "anything wearable": "wearable", "things to wear": "wearable",
    "clothes": "clothing", "apparel": "clothing", "garments": "clothing",
    "jewellery": "jewelry",
    "edible": "food", "edibles": "food", "food items": "food",
    "home decor": "decor", "decorations": "decor",
    "plush": "stuffed animals", "plush toys": "stuffed animals", "plushies": "stuffed animals",
    "stuffed toys": "stuffed animals", "stuffed animal": "stuffed animals",
    "scented products": "scented", "scented items": "scented", "fragrances": "scented",
    "electronics": "electronic", "battery": "electronic", "batteries": "electronic",
    "battery operated": "electronic", "battery-operated": "electronic",
}


def exclusion_group(term):
    """Canonical group for a user term ("plush toys" -> "stuffed animals") or None."""
    key = " ".join(str(term).lower().split())
    if key in EXCLUSION_GROUPS:
        return key
    return EXCLUSION_ALIASES.get(key)


def normalize_exclusions(exclusions):
    """What the filter should receive, in order and without duplicates."""
    out = []
    for term in exclusions:
        group = exclusion_group(term)
        if group in FILTER_GROUPS:
            terms = [group]
        else:
            terms = [term, *EXPANDED_GROUPS.get(group, [])]
        out += [t for t in terms if t not in out]
    return out
