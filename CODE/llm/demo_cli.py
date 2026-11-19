"""Command-line demo of the full Gift Concierge loop on the development sample.

    # interactive, real LLM (needs .env with FOUNDRY_* vars)
    python CODE/llm/demo_cli.py

    # no credentials: rule-based stand-in for the LLM
    python CODE/llm/demo_cli.py --offline

    # scripted turns, with state changes / query / ranking shown
    python CODE/llm/demo_cli.py --offline --verbose \
        --turns "gift for my 8 year old nephew" "he loves dinosaurs and space" \
                "under $30" "cheaper" "actually he's into lego instead"

Uses the stub ranker (CODE/llm/stub_ranker.py) until the real ranking stage is
ready. Builds the sample index first if it does not exist yet.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.conversation import GiftConcierge  # noqa: E402
from retrieval.config import INDEX_FILE, REPO_ROOT, SAMPLE_CORPUS, SAMPLE_INDEX_DIR  # noqa: E402
from retrieval.corpus import load_products  # noqa: E402


def ensure_sample_index(index_dir, corpus):
    if (Path(index_dir) / INDEX_FILE).exists():
        return
    from retrieval.build_faiss_index import build_index
    from retrieval.embed_corpus import embed_corpus
    print(f"Building sample index in {index_dir} (one-time, ~1 min)...")
    embed_corpus(corpus, index_dir, log=lambda *a: None)
    build_index(index_dir, log=lambda *a: None)


def show_details(response, products):
    print("  [constraints]", json.dumps(
        {k: v for k, v in response["constraints"].items() if v not in (None, [], False)}))
    for change in response["state_changes"]:
        print(f"  [state] {change['field']}: {change['old']!r} -> {change['new']!r} "
              f"({change['rule']})")
    for conflict in response["conflicts"]:
        print(f"  [conflict] {conflict['kind']}")
    if response.get("extraction_error"):
        print(f"  [extraction error] {response['extraction_error']}")
    if "query_text" in response:
        print(f"  [query] {response['query_text']}")
        top = response["retrieved"][:5]
        print("  [semantic top5]", ", ".join(
            f"{h['parent_asin']}({h['score']:.3f})" for h in top))
        print(f"  [stub ranker] kept {len(response['ranked'])} of {len(response['retrieved'])}")
    for item in response.get("explanations", []):
        print(f"  [evidence #{item['rank']}] {item['evidence_ids']}"
              + (f" invalid={item['invalid_citations']}" if item["invalid_citations"] else "")
              + (" (fallback)" if item["fallback"] else ""))


def main():
    parser = argparse.ArgumentParser(description="Gift Concierge CLI demo")
    parser.add_argument("--offline", action="store_true",
                        help="use the rule-based demo extractor instead of the LLM")
    parser.add_argument("--turns", nargs="+", help="scripted user messages (non-interactive)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--corpus", default=str(SAMPLE_CORPUS))
    parser.add_argument("--index-dir", default=str(SAMPLE_INDEX_DIR))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log", default=str(REPO_ROOT / "logs" / "state_changes.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.offline:
        from llm.offline_llm import OfflineLLM
        llm = OfflineLLM()
    else:
        from llm.llm_client import LLMClient, LLMConfigError
        try:
            llm = LLMClient.from_env()
        except LLMConfigError as exc:
            sys.exit(f"{exc}\n(Or run with --offline.)")

    ensure_sample_index(args.index_dir, args.corpus)
    from retrieval.semantic_search import SemanticRetriever
    retriever = SemanticRetriever(args.index_dir)
    products = load_products(args.corpus)
    bot = GiftConcierge(llm, retriever, products, top_n=args.top_n, top_k=args.top_k,
                        log_path=args.log)

    print("Gift Concierge demo (sample corpus, stub ranker). Ctrl+C or 'quit' to exit.")
    print("Bot: Hi! Who are you shopping for?")

    turns = iter(args.turns) if args.turns else None
    while True:
        try:
            if turns is not None:
                message = next(turns)
                print(f"\nYou: {message}")
            else:
                message = input("\nYou: ").strip()
        except (StopIteration, EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break
        response = bot.handle(message)
        if args.verbose:
            show_details(response, products)
        print(f"Bot: {response['message']}")

    print(f"\nState change log: {args.log}")


if __name__ == "__main__":
    main()
