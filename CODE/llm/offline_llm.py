"""DEMO-ONLY rule-based stand-in for the LLM (`demo_cli.py --offline`).

Lets the whole loop run without Foundry credentials. It understands a small
set of phrasings with regexes and is NOT used for evaluation; real extraction
and explanations go through LLMClient.
"""

import re

from llm.constraints import TURN_DELTA_SCHEMA

OCCASIONS = ["birthday", "christmas", "graduation", "anniversary", "wedding",
             "housewarming", "baby shower", "mother's day", "father's day",
             "valentine's day", "retirement", "holiday"]
STOP = r"(?=,|\.|!|\?|$| who\b| that\b| and (?:is|she|he|they)\b| under\b| below\b| around\b| for\b| but\b| instead\b)"
MONEY = r"\$\s?(\d+(?:\.\d+)?)"


def _terms(text):
    parts = re.split(r",|\band\b|\bor\b", text)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def extract(message):
    m = message.lower()
    d = {name: None for name in TURN_DELTA_SCHEMA["properties"]}
    d.update(interests_add=[], interest_labels=[], interests_remove=[], exclusions_add=[],
             replace_interests=False, no_budget_preference=False)

    if re.search(r"no (budget|limit)|budget (doesn't|does not) matter|any price", m):
        d["no_budget_preference"] = True
    elif r := re.search(rf"between {MONEY} and {MONEY}|{MONEY}\s?(?:-|to)\s?{MONEY}", m):
        lo, hi = [float(x) for x in r.groups() if x]
        d["budget_min"], d["budget_max"] = lo, hi
    elif re.search(rf"(?:under|below|less than|up to|max|at most|at least|over|above|more than) {MONEY}", m):
        if r := re.search(rf"(?:under|below|less than|up to|max|at most) {MONEY}", m):
            d["budget_max"] = float(r.group(1))
        if r := re.search(rf"(?:at least|over|above|more than) {MONEY}", m):
            d["budget_min"] = float(r.group(1))
    elif r := re.search(rf"(?:around|about) {MONEY}", m):
        d["budget_min"], d["budget_max"] = float(r.group(1)) * 0.8, float(r.group(1)) * 1.2
    if re.search(r"cheaper|less expensive", m):
        d["budget_relative"] = "cheaper"
    elif re.search(r"more expensive|pricier|nicer", m):
        d["budget_relative"] = "more_expensive"
    if d["budget_relative"]:
        if r := re.search(rf"{MONEY} (?:cheaper|less|more)", m):
            d["budget_relative_amount"] = float(r.group(1))
    elif d["budget_min"] is None and d["budget_max"] is None and not d["no_budget_preference"]:
        if r := re.search(MONEY, m):
            d["budget_max"] = float(r.group(1))

    if r := re.search(r"for (?:my|our) ((?:\d+[- ]year[- ]old )?[a-z\- ]+?)" + STOP, m):
        d["recipient"] = "my " + r.group(1).strip()
    if r := re.search(r"(\d+)[- ](?:year|yr)s?[- ]old|\bage (\d+)|\b(?:she|he)'s (\d+)\b", m):
        d["age"] = float(next(g for g in r.groups() if g))
    for occasion in OCCASIONS:
        if occasion in m:
            d["occasion"] = occasion
            break

    if r := re.search(r"(?:not into|no longer likes?|doesn't like|does not like) ([a-z\- ,]+?)(?: anymore|" + STOP + ")", m):
        d["interests_remove"] = _terms(r.group(1))
    for r in re.finditer(r"(?:likes|loves|into|interested in|fan of|enjoys) ([a-z\- ,&']+?)" + STOP, m):
        if re.search(r"(not|n't|no longer)\s*$", m[: r.start()].rstrip()[-12:]):
            continue  # negated: handled as interests_remove above
        d["interests_add"] += [t for t in _terms(r.group(1)) if t not in d["interests_add"]]
    if "instead" in m and d["interests_add"]:
        d["replace_interests"] = True
    for r in re.finditer(r"\bno ([a-z\- ]+?)(?=,|\.|$| and | but | please)", m):
        if not re.match(r"(budget|limit|longer)", r.group(1)):
            d["exclusions_add"].append(r.group(1).strip())

    if re.search(r"\bnothing (?:wearable|to wear)\b", m):
        d["exclusions_add"].append("wearable")

    for word, category in (("handmade", "Handmade Products"), ("book", "Books"),
                           ("toy", "Toys & Games"), ("game", "Toys & Games")):
        if re.search(rf"(?<!no )\b{word}", m) and not any(word in e for e in d["exclusions_add"]):
            d["category_preference"] = category
            break
    return d


def explain(messages):
    """Template explanations citing title, rating and first feature/category."""
    prompt = messages[-1]["content"]
    out = []
    for block in prompt.split("\n\nProduct ")[1:]:
        pid = block.split(":", 1)[0].strip()
        ids = re.findall(r"^\[(P\d+\.[^\]]+\]?)\] ", block, flags=re.M)
        lines = dict(re.findall(r"^\[(P\d+\.[^\]]+\]?)\] (.*)$", block, flags=re.M))
        cite, parts = [], []
        if f"{pid}.rating" in lines:
            cite.append(f"{pid}.rating")
            parts.append(lines[f"{pid}.rating"])
        extra = next((i for i in ids if ".feature[" in i or ".category" in i), None)
        if extra:
            cite.append(extra)
            parts.append(lines[extra].rstrip("."))
        text = "; ".join(parts) + "." if parts else "Matches your search."
        out.append({"product_id": pid, "explanation": "[offline] " + text, "evidence_ids": cite})
    return {"explanations": out}


class OfflineLLM:
    def complete(self, prompt_or_messages, system=None):
        return ""

    def complete_json(self, prompt_or_messages, schema, name, system=None):
        messages = prompt_or_messages
        if name == "gift_constraints_delta":
            text = messages[-1]["content"].split("Latest user message:\n", 1)[-1]
            return extract(text)
        if name == "gift_explanations":
            return explain(messages)
        raise ValueError(f"OfflineLLM has no handler for {name}")
