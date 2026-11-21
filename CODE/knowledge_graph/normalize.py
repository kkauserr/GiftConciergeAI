"""Normalisation of raw corpus values before they become KG nodes.

Each function documents the sample-verified problem it handles. Normalisers
return None (not a guess) when a value is unusable, so extractors can skip
the edge rather than fabricate one.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

REPLACEMENT_CHAR = "�"

# Values that mean "no real brand". Creating a Brand node for these would
# link unrelated products ("Generic" covers 86/300 Handmade sample products).
PLACEHOLDER_BRANDS = {
    "generic", "unknown", "unbranded", "no brand", "none", "n/a", "na",
    "null", "not applicable", "various", "other", "brand", "-",
}

# Scraping artefact seen in the sample: "Visit the Funko Store" -> "Funko".
_VISIT_STORE = re.compile(r"^\s*visit\s+the\s+(.+?)\s+store\s*$", re.IGNORECASE)

# Contributor role suffixes found in Books `store` values,
# e.g. "Sandra Wilde (Editor)", "Charles Platt (Author)".
_ROLE_SUFFIX = re.compile(
    r"\s*\((author|editor|illustrator|translator|contributor|narrator|"
    r"foreword|introduction|compiler|photographer)[^)]*\)\s*$",
    re.IGNORECASE,
)

# Maps the corpus `domain` value to the root name used in category paths.
DOMAIN_ROOT_CATEGORY = {
    "Books": "Books",
    "Handmade Products": "Handmade Products",
    "Toys & Games": "Toys & Games",
}


def clean_text(value: Any) -> Optional[str]:
    """Unicode-normalise, collapse whitespace; None for empty/non-string."""
    if not isinstance(value, str):
        return None
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    return value or None


def has_encoding_issue(value: Optional[str]) -> bool:
    """True if the text contains U+FFFD, meaning a character was lost during
    upstream decoding and cannot be recovered. In the sample this occurs only
    in one product's description; no title or category is affected."""
    return bool(value) and REPLACEMENT_CHAR in value


def is_placeholder_brand(name: Optional[str]) -> bool:
    return name is None or name.strip().casefold() in PLACEHOLDER_BRANDS


def strip_role_suffix(name: str) -> str:
    return _ROLE_SUFFIX.sub("", name).strip()


def normalize_brand_name(raw: Any) -> Optional[str]:
    """Clean a brand/author string. Returns None for empty or placeholder
    values. Display casing is preserved; ID-level case folding happens in
    ontology.slugify, so "LEGO" and "Lego" map to the same node."""
    name = clean_text(raw)
    if name is None:
        return None
    m = _VISIT_STORE.match(name)
    if m:
        name = m.group(1).strip()
    name = strip_role_suffix(name)
    if not name or is_placeholder_brand(name):
        return None
    return name


def resolve_brand(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """Choose the brand/author for a product.

    Returns (name, source_field, status). status is one of:
      "ok"          - usable name found
      "placeholder" - primary value is a placeholder such as "Generic";
                      deliberately NO fallback, because secondary fields on
                      these records are also unreliable (one sample "Generic"
                      product lists Manufacturer = "Chainsaw Man", a licence)
      "missing"     - no value in any field

    Order: brand_or_author, then details.Brand, then details.Manufacturer.
    `store` is not used as a fallback: for Books it holds multi-contributor
    role strings and sometimes a different person (an editor), and
    preprocessing already used it when building brand_or_author.
    """
    primary_raw = clean_text(record.get("brand_or_author"))
    if primary_raw is not None:
        name = normalize_brand_name(primary_raw)
        if name:
            return name, "brand_or_author", "ok"
        return None, "brand_or_author", "placeholder"

    details = record.get("details") or {}
    for key in ("Brand", "Manufacturer"):
        raw = clean_text(details.get(key))
        if raw is None:
            continue
        name = normalize_brand_name(raw)
        if name:
            return name, f"details.{key}", "ok"
        return None, f"details.{key}", "placeholder"
    return None, None, "missing"


def brand_kind(domain: Optional[str], source_field: Optional[str]) -> str:
    """Books' brand_or_author comes from the structured author object."""
    if domain == "Books" and source_field == "brand_or_author":
        return "author"
    return "brand"


def normalize_category_path(record: Dict[str, Any]) -> Tuple[List[str], str]:
    """Return (path, source_field).

    Uses `categories` when non-empty. Otherwise falls back to the domain's root
    category, derived from the verified `domain` field. `main_category` is not
    used, because it contains values like "Buy a Kindle" and "AMAZON FASHION"
    that are not part of the category hierarchy.

    Paths are kept as given, even when the root differs from the domain (four
    sample Toys products are rooted at "Remote & App Controlled Vehicles &
    Parts"). Re-rooting them would invent hierarchy that the data does not
    have. Consecutive duplicate names are collapsed.
    """
    raw = record.get("categories") or []
    path: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            name = clean_text(item)
            if name and (not path or path[-1] != name):
                path.append(name)
    if path:
        return path, "categories"
    root = DOMAIN_ROOT_CATEGORY.get(record.get("domain"))
    return ([root], "domain") if root else ([], "none")


# Recipient and occasion words. These belong to the conversation state. They
# are used only to detect SEO keyword lists, never to create KG entities.
RECIPIENT_OCCASION_TERMS = (
    "him", "her", "men", "women", "man", "woman", "mom", "mother", "dad", "father",
    "boyfriend", "girlfriend", "husband", "wife", "friend", "friends", "best friend",
    "kids", "kid", "boys", "girls", "teens", "teen", "toddler", "baby", "grandma",
    "grandpa", "grandmother", "grandfather", "sister", "brother", "daughter", "son",
    "coworker", "coworkers", "teacher", "teachers", "nurse", "nurses", "boss",
    "christmas", "xmas", "birthday", "birthdays", "anniversary", "valentine",
    "valentines", "valentine's day", "mother's day", "father's day", "graduation",
    "wedding", "holiday", "holidays", "halloween", "thanksgiving", "easter",
    "retirement", "housewarming", "baby shower", "bridal shower", "stocking stuffer",
    "stocking stuffers",
)
_RO_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(sorted((re.escape(t) for t in RECIPIENT_OCCASION_TERMS),
                                     key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_LIST_SEPARATORS = re.compile(r"[,;|/]")


def keyword_list_items(text: Optional[str], min_segments: int = 6,
                       min_recipient_segments: int = 2,
                       max_item_words: int = 4) -> List[Tuple[int, int]]:
    """Spans of the short items in an SEO keyword list. Returns [] when the
    text is not a keyword list.

    Example from the sample: "...boyfriend, him, kids, teens, best friend,
    cat lover, Christmas, Xmas, holiday...". "cat lover" here is a search
    keyword, not evidence of who the product suits.

    A text counts as a keyword list when either:
      (a) it has at least `min_segments` separator-delimited segments and at
          least `min_recipient_segments` of them contain a recipient or
          occasion term, or
      (b) it has at least `long_list_segments` segments and at least 75% of
          them are short. These are lists of compatible uses or available
          designs, e.g. a sticker for "laptops, guitars, luggage,
          refrigerators, ..." or "religious, Christian, funny sayings,
          nautical, ...". Those items say what the product can go with,
          not what it is.
    Only short segments (at most `max_item_words` words) are returned, so a
    real sentence before or after the list is still usable.
    """
    if not text:
        return []
    segments, start = [], 0
    for m in _LIST_SEPARATORS.finditer(text):
        segments.append((start, m.start()))
        start = m.end()
    segments.append((start, len(text)))
    runs = short_item_runs(text)
    if len(segments) < min_segments:
        return runs
    short = [(s, e) for s, e in segments if len(text[s:e].split()) <= max_item_words]
    ro = sum(1 for s, e in segments if _RO_PATTERN.search(text[s:e]))
    is_recipient_list = ro >= min_recipient_segments
    is_long_list = len(segments) >= LONG_LIST_SEGMENTS and len(short) >= 0.75 * len(segments)
    return short if (is_recipient_list or is_long_list) else runs


LONG_LIST_SEGMENTS = 8

# Rule (c) of keyword_list_items: a run of consecutive very short items.
# " - " and "~" also separate items here, because titles use them as list
# separators ("... Resin - Small Cast Jewelry, Fishing Lures, Dolls, Hobby,
# RC, Crafts ~ Cures in Mins").
_RUN_SEPARATORS = re.compile(r"[,;|/~]|\s-\s")
SHORT_RUN_MIN_ITEMS = 5
SHORT_RUN_MAX_WORDS = 2


def short_item_runs(text: Optional[str], min_items: int = SHORT_RUN_MIN_ITEMS,
                    max_words: int = SHORT_RUN_MAX_WORDS) -> List[Tuple[int, int]]:
    """Spans of items in a run of at least `min_items` consecutive segments
    of at most `max_words` words. Such runs list uses or compatible items
    ("Jewelry, Fishing Lures, Dolls, Hobby, RC, Crafts"), so a single item
    says nothing about what the product is."""
    if not text:
        return []
    segments, start = [], 0
    for m in _RUN_SEPARATORS.finditer(text):
        segments.append((start, m.start()))
        start = m.end()
    segments.append((start, len(text)))
    out: List[Tuple[int, int]] = []
    run: List[Tuple[int, int]] = []
    for s, e in segments + [(len(text), len(text))]:
        if 0 < len(text[s:e].split()) <= max_words:
            run.append((s, e))
            continue
        if len(run) >= min_items:
            out.extend(run)
        run = []
    return out


# Compatibility clauses say what the product fits, not what it is: "Card
# Binder ... Fit for TCG Yugioh card and Sports Cards", "Replacement Plug
# compatible with ...". The span runs from the marker to the clause end.
COMPATIBILITY_MARKERS = ("fit for", "fits for", "fits", "compatible with", "compatible for",
                         "for use with", "works with", "designed to fit")
_COMPAT = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(m) for m in sorted(COMPATIBILITY_MARKERS, key=len, reverse=True))
    + r")(?!\w)", re.IGNORECASE)
_COMPAT_END = re.compile(r"[.;|()\[\]【】!?]|,\s|\s[-–—]\s")


def compatibility_spans(text: Optional[str]) -> List[Tuple[int, int]]:
    if not text:
        return []
    spans = []
    for m in _COMPAT.finditer(text):
        end = _COMPAT_END.search(text, m.end())
        spans.append((m.start(), end.start() if end else len(text)))
    return spans


# Cross-sell sentences point at *other* products ("You Can Consider This Cat
# Puzzle", "Check out our other designs"). Text in them is never evidence
# about the listed product.
CROSS_SELL_MARKERS = (
    "consider this", "consider our", "check out our", "check our", "also check",
    "also available", "also like", "you may also", "you might also", "see our",
    "visit our", "browse our", "our other", "other designs", "other listings",
    "more designs", "search for", "click on", "matching set available",
)
_CROSS_SELL = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(m) for m in CROSS_SELL_MARKERS) + r")(?!\w)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[.!?;|]\s+|[【】]")


def cross_sell_spans(text: Optional[str]) -> List[Tuple[int, int]]:
    """Spans of the sentences in `text` that contain a cross-sell marker."""
    if not text or not _CROSS_SELL.search(text):
        return []
    spans, start = [], 0
    for m in list(_SENTENCE_BREAK.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        if _CROSS_SELL.search(text[start:end]):
            spans.append((start, end))
        if m:
            start = m.end()
    return spans


# A contents list names several counted items: "1 Plastic Funnel, 1 Wooden
# Scratch Stick, 1 Plastic Flask" or "12 wood pencils 16 notebooks and 12
# rulers". A material inside one of those items describes a component of a
# kit, not the product. Measurements ("110 lb card stock", "18 inch chain")
# and purities ("925 sterling silver", "14k gold") are not item counts.
_UNIT_WORDS = {"inch", "inches", "in", "cm", "mm", "m", "ft", "feet", "oz", "lb", "lbs",
               "pt", "gsm", "mil", "g", "kg", "ml", "l", "k", "kt", "ct", "pound",
               "pounds", "ounce", "ounces", "gauge", "%"}
_COUNTED_ITEM = re.compile(
    r"(?<![\w.])(\d{1,3})\s*(?:x\s+|pcs?\s*|pieces?\s*|pack\s*)?"
    r"(?=[A-Za-z])([A-Za-z][\w'-]*)"
)
CONTENTS_MIN_ITEMS = 3


def contents_list_items(text: Optional[str],
                        min_items: int = CONTENTS_MIN_ITEMS) -> List[Tuple[int, int]]:
    """Spans of counted items when `text` lists at least `min_items` of them.
    Each span runs from the count to the next separator or count."""
    if not text:
        return []
    starts = [m for m in _COUNTED_ITEM.finditer(text)
              if m.group(2).lower() not in _UNIT_WORDS and m.group(1) != "925"]
    if len(starts) < min_items:
        return []
    spans = []
    for i, m in enumerate(starts):
        limit = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        brk = re.search(r"[,;.()\[\]【】]|and", text[m.end():limit])
        spans.append((m.start(), m.end() + brk.start() if brk else limit))
    return spans


def age_range_valid(age_min: Any, age_max: Any) -> Optional[bool]:
    """False when age_min > age_max, None when either bound is missing.

    The upstream mixed-unit bug ("36 months - 7 years" read as 36-7 years)
    was fixed in CODE/data/preprocess_products.py (commit 3a62d51); the
    regenerated sample has no inverted ranges. The check stays as a guard
    for future corpora. Values are never rewritten here."""
    if isinstance(age_min, (int, float)) and isinstance(age_max, (int, float)):
        return age_min <= age_max
    return None
