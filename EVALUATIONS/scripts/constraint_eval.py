import re
from llm.llm_client import LLMClient
from llm.constraints import extract_turn

tests = [
    {
        "text": "Birthday gift for my 8 year old nephew who loves dinosaurs, under $40",
        "gold": {
            "recipient": "nephew",
            "occasion": "birthday",
            "interests": {"dinosaurs"},
            "budget_max": 40,
            "age": 8,
        },
    },
    {
        "text": "Graduation gift for my sister who likes coffee and reading, under $80",
        "gold": {
            "recipient": "sister",
            "occasion": "graduation",
            "interests": {"coffee", "reading"},
            "budget_max": 80,
        },
    },
    {
        "text": "A Christmas gift for my dad who enjoys golf, no more than $50",
        "gold": {
            "recipient": "dad",
            "occasion": "christmas",
            "interests": {"golf"},
            "budget_max": 50,
        },
    },
    {
        "text": "Gift for my 6 year old cousin who loves airplanes, under $30",
        "gold": {
            "recipient": "cousin",
            "interests": {"airplanes"},
            "budget_max": 30,
            "age": 6,
        },
    },
    {
        "text": "Housewarming gift for a friend who likes plants and home decor, under $40",
        "gold": {
            "recipient": "friend",
            "occasion": "housewarming",
            "interests": {"plants", "home decor"},
            "budget_max": 40,
        },
    },
    {
        "text": "Gift for my mom who likes mystery books, under $25",
        "gold": {
            "recipient": "mom",
            "interests": {"mystery books"},
            "budget_max": 25,
        },
    },
    {
        "text": "Anniversary gift for my wife who likes handmade jewelry, between $30 and $70",
        "gold": {
            "recipient": "wife",
            "occasion": "anniversary",
            "interests": {"handmade jewelry"},
            "budget_min": 30,
            "budget_max": 70,
        },
    },
    {
        "text": "Gift for my teenage brother who likes cars, under $35, no books",
        "gold": {
            "recipient": "brother",
            "interests": {"cars"},
            "budget_max": 35,
            "exclusions": {"books"},
        },
    },
    {
        "text": "A Valentine's Day gift for my girlfriend who loves baking and chocolate, under $60",
        "gold": {
            "recipient": "girlfriend",
            "occasion": "valentine's day",
            "interests": {"baking", "chocolate"},
            "budget_max": 60,
        },
    },
    {
        "text": "Birthday gift for my 10 year old niece who likes art, under $25, no dolls",
        "gold": {
            "recipient": "niece",
            "occasion": "birthday",
            "interests": {"art"},
            "budget_max": 25,
            "age": 10,
            "exclusions": {"dolls"},
        },
    },
]

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def setify(v):
    if v is None:
        return set()
    if isinstance(v, (list, set, tuple)):
        return {norm(x) for x in v}
    return {norm(v)}

def scalar_match(field, pred, gold):
    if field in {"budget_min", "budget_max", "age"}:
        try:
            return float(pred) == float(gold)
        except (TypeError, ValueError):
            return False

    p = norm(pred)
    g = norm(gold)

    if field == "recipient":
        # "my 8 year old nephew" should match "nephew"
        return g == p or g in p.split()

    return p == g

llm = LLMClient.from_env()

tp = fp = fn = 0

for i, t in enumerate(tests, 1):
    r = extract_turn(llm, t["text"])
    d = r.delta.model_dump()

    pred = {
        "recipient": d.get("recipient"),
        "occasion": d.get("occasion"),
        "interests": setify(d.get("interests_add")),
        "budget_min": d.get("budget_min"),
        "budget_max": d.get("budget_max"),
        "age": d.get("age"),
        "exclusions": setify(d.get("exclusions_add")),
    }

    gold = t["gold"]

    for field in ["recipient", "occasion", "budget_min", "budget_max", "age"]:
        g = gold.get(field)
        p = pred.get(field)

        if g is not None:
            if p is not None and scalar_match(field, p, g):
                tp += 1
            else:
                fn += 1
                if p is not None:
                    fp += 1
        elif p is not None:
            fp += 1

    for field in ["interests", "exclusions"]:
        g = setify(gold.get(field))
        p = setify(pred.get(field))

        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    print(f"\n{i}. {t['text']}")
    print("pred:", pred)
    print("gold:", gold)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\n--- Constraint Extraction Evaluation ---")
print("TP:", tp, "FP:", fp, "FN:", fn)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")