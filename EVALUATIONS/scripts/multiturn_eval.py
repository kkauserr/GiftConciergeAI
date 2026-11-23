from llm.llm_client import LLMClient
from llm.constraints import extract_turn
from llm.dialogue_state import DialogueState

tests = [
    {
        "name": "cheaper preserves recipient and interest",
        "turns": [
            "Gift for my sister who likes coffee, under $50",
            "Make it cheaper"
        ],
        "expected": {
            "recipient_contains": "sister",
            "interests": {"coffee"},
            "budget_max": 37.5
        }
    },
    {
        "name": "explicit budget update preserves other fields",
        "turns": [
            "Birthday gift for my nephew who likes dinosaurs, under $40",
            "Actually make the budget $60"
        ],
        "expected": {
            "recipient_contains": "nephew",
            "occasion": "birthday",
            "interests": {"dinosaurs"},
            "budget_max": 60
        }
    },
    {
        "name": "new interest is added",
        "turns": [
            "Gift for my friend who likes hiking, under $50",
            "They also like photography"
        ],
        "expected": {
            "recipient_contains": "friend",
            "interests": {"hiking", "photography"},
            "budget_max": 50
        }
    },
    {
        "name": "exclusion added without losing earlier state",
        "turns": [
            "Gift for my mom who likes mystery books, under $30",
            "No hardcover books"
        ],
        "expected": {
            "recipient_contains": "mom",
            "interests": {"mystery books"},
            "budget_max": 30,
            "exclusions_contains": "hardcover books"
        }
    },
    {
        "name": "occasion change preserves recipient and interests",
        "turns": [
            "Gift for my dad who likes golf, under $50",
            "This is actually for Christmas"
        ],
        "expected": {
            "recipient_contains": "dad",
            "occasion": "christmas",
            "interests": {"golf"},
            "budget_max": 50
        }
    }
]

def norm(x):
    return str(x).strip().lower() if x is not None else None

llm = LLMClient.from_env()

passed = 0

for test in tests:
    state = DialogueState()

    for message in test["turns"]:
        result = extract_turn(llm, message, state.constraints, None)
        state.apply_delta(result.delta, message)

    c = state.constraints
    exp = test["expected"]

    checks = []

    if "recipient_contains" in exp:
        checks.append(exp["recipient_contains"] in norm(c.recipient))

    if "occasion" in exp:
        checks.append(norm(c.occasion) == norm(exp["occasion"]))

    if "interests" in exp:
        pred_interests = {norm(x) for x in c.interests}
        checks.append(exp["interests"] == pred_interests)

    if "budget_max" in exp:
        checks.append(float(c.budget_max) == float(exp["budget_max"]))

    if "exclusions_contains" in exp:
        pred_exclusions = {norm(x) for x in c.exclusions}
        checks.append(norm(exp["exclusions_contains"]) in pred_exclusions)

    ok = all(checks)

    print(f"\n{test['name']}")
    print("Final state:", c.model_dump())
    print("PASS" if ok else "FAIL")

    if ok:
        passed += 1

accuracy = passed / len(tests)

print("\n--- Multi-turn Consistency Evaluation ---")
print(f"Passed: {passed}/{len(tests)}")
print(f"Consistency: {accuracy:.3f}")