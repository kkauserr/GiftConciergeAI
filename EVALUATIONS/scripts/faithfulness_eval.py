from llm.llm_client import LLMClient
from llm.conversation import GiftConcierge
from llm.ranking_adapter import HybridRanker
from retrieval.semantic_search import SemanticRetriever
from retrieval.corpus import load_products
from ranking.pipeline import GiftRanker

CORPUS = "/scratch/jkauser/gift-concierge/data/dev_subset_v2.jsonl"
SEMANTIC_INDEX = "/scratch/jkauser/gift-concierge/indexes/semantic-dev60k"
RANK_INDEX = "/scratch/jkauser/gift-concierge/indexes/dev60k"

queries = [
    "Birthday gift for my 8 year old nephew who loves dinosaurs, under $40",
    "Graduation gift for my sister who likes coffee and reading, under $80",
    "Christmas gift for my dad who enjoys golf, under $50",
    "Housewarming gift for a friend who likes plants and home decor, under $40",
    "Birthday gift for my 10 year old niece who likes art, under $25"
]

print("Loading models and indexes...")

llm = LLMClient.from_env()
retriever = SemanticRetriever(SEMANTIC_INDEX)
products = load_products(CORPUS)
ranker = HybridRanker(GiftRanker(RANK_INDEX, load_tfidf=False))

total = 0
valid = 0
with_evidence = 0
fallbacks = 0
invalid_count = 0

for i, query in enumerate(queries, 1):
    bot = GiftConcierge(
        llm,
        retriever,
        products,
        ranker=ranker,
        top_n=200,
        top_k=3
    )

    result = bot.handle(query)

    print(f"\n=== Query {i} ===")
    print(query)

    if result["type"] != "recommendations":
        print("No recommendations:", result["message"])
        continue

    for exp in result["explanations"]:
        total += 1

        invalid = exp["invalid_citations"]
        evidence = exp["used_evidence"]

        if not invalid:
            valid += 1

        if evidence:
            with_evidence += 1

        if exp["fallback"]:
            fallbacks += 1

        invalid_count += len(invalid)

        print("\nExplanation:", exp["explanation"])
        print("Evidence IDs:", exp["evidence_ids"])
        print("Invalid citations:", invalid)

citation_validity = valid / total if total else 0
evidence_coverage = with_evidence / total if total else 0

print("\n--- Explanation Faithfulness Evaluation ---")
print(f"Explanations evaluated: {total}")
print(f"Valid-citation explanations: {valid}/{total}")
print(f"Citation validity: {citation_validity:.3f}")
print(f"Evidence coverage: {evidence_coverage:.3f}")
print(f"Invalid citations total: {invalid_count}")
print(f"Fallback explanations: {fallbacks}")