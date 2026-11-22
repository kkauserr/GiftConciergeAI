"""Map free-text interests to the knowledge graph's shared interest labels.

The label set is knowledge_graph/vocab/interests.json (owner: Jenna). It is
read from that file, so there is one source of truth and no graph build (~15 s)
just to list labels. Without the file no labels are sent.

Free-text interests stay the retrieval query ("star wars"); labels
("scifi_fantasy") only go to the KG as query_interests.
"""

import json
from functools import lru_cache
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parents[1] / "knowledge_graph" / "vocab" / "interests.json"


def _key(text):
    words = " ".join(str(text).lower().replace("-", " ").replace("_", " ").split())
    return words[:-1] if words.endswith("s") and not words.endswith("ss") else words


@lru_cache(maxsize=1)
def vocabulary():
    if not VOCAB_PATH.exists():
        return {"version": None, "interests": []}
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return {"version": f"interests@{data['version']}", "interests": data["interests"]}


def interest_labels():
    return sorted(i["label"] for i in vocabulary()["interests"])


@lru_cache(maxsize=1)
def _alias_map():
    out = {}
    items = vocabulary()["interests"]
    for item in items:
        for phrase in [item["label"], item.get("display_name"), *(item.get("aliases") or [])]:
            if phrase:
                out.setdefault(_key(phrase), item["label"])
    # "Coffee & Tea" also answers to "coffee" and to "tea". Added last so an
    # explicit alias always wins.
    for item in items:
        parts = [p.strip() for p in (item.get("display_name") or "").split("&")]
        if len(parts) == 2:
            for part in parts:
                out.setdefault(_key(part), item["label"])
    return out


def label_for(interest):
    """Label by label / display name / alias match, or None. Never guesses."""
    key = _key(interest)
    return _alias_map().get(key) or _alias_map().get(
        _key(key.removesuffix(" lover").removesuffix(" fan")))


def labels_for(interests, llm_labels=None):
    """KG labels for the current interests, in order.

    llm_labels ({interest: label} from extraction) wins over the alias lookup.
    Labels outside the vocabulary are dropped.
    """
    valid = set(interest_labels())
    out = []
    for interest in interests:
        label = (llm_labels or {}).get(interest)
        if label not in valid:
            label = label_for(interest)
        if label and label not in out:
            out.append(label)
    return out


def prompt_hints():
    """One 'label: example phrasings' line per label for the extraction prompt."""
    return "\n".join(f"  {i['label']}: {', '.join((i.get('aliases') or [])[:4])}"
                     for i in vocabulary()["interests"])
