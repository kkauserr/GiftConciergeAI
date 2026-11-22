"""Builds the recommendation stack once and hands out one GiftConcierge per conversation.

Mirrors how CODE/llm/demo_cli.py (Shashank) wires the pieces, so the backend
runs exactly the pipeline the LLM layer was written for:

    LLM          llm.llm_client.LLMClient (Foundry) or llm.offline_llm.OfflineLLM
    retrieval    retrieval.semantic_search.SemanticRetriever (BGE + FAISS)
    ranking      CODE.ranking.pipeline.GiftRanker via llm.ranking_adapter.HybridRanker
                 (Disha), or the LLM layer's stub ranker when no BM25 index exists
    evidence     knowledge_graph.api.load_kg via llm.kg_adapter.KGEvidence (Jenna)

Heavy objects (models, indexes, products, KG) are shared across conversations.
Each conversation only owns its GiftConcierge, i.e. its DialogueState.

Nothing here makes recommendation decisions; it only connects components.
"""

import logging
import sys
import time

from .config import CODE_DIR, REPO_ROOT, Settings

log = logging.getLogger("gift_concierge.engine")

# The LLM / retrieval / KG packages import each other as top-level modules
# ("from llm import ...", "import knowledge_graph"), and the ranking package as
# "CODE.ranking". Both roots must be importable.
for root in (CODE_DIR, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class EngineUnavailable(RuntimeError):
    """The recommendation stack could not be started (missing component or config)."""


class TfidfStandInRetriever:
    """DEV-ONLY stand-in for SemanticRetriever when FAISS / sentence-transformers
    or the embedding index are not available. Same interface and output shape.
    Reported by /health so it is never mistaken for the real semantic retrieval."""

    def __init__(self, products):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.asins = list(products)
        texts = [products[a].get("embed_text") or products[a].get("title") or "" for a in self.asins]
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query_text, top_n=100):
        import numpy as np

        scores = (self.matrix @ self.vectorizer.transform([query_text]).T).toarray().ravel()
        order = np.argsort(-scores)[:top_n]
        return [{"parent_asin": self.asins[i], "score": round(float(scores[i]), 6), "rank": r}
                for r, i in enumerate(order, 1) if scores[i] > 0]


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.status = {}
        started = time.perf_counter()

        try:
            from llm.conversation import GiftConcierge
            from retrieval.corpus import load_products
        except ImportError as exc:
            raise EngineUnavailable(
                "The LLM/conversation layer (CODE/llm, CODE/retrieval) is not in this checkout. "
                "Merge the shashank-g/semantic-llm branch first."
            ) from exc
        self._GiftConcierge = GiftConcierge

        if not settings.corpus.exists():
            raise EngineUnavailable(f"Product corpus not found: {settings.corpus}")
        self.products = load_products(settings.corpus)
        self.status["products"] = len(self.products)
        self.status["corpus"] = str(settings.corpus)

        self.llm = self._build_llm()
        self.retriever = self._build_retriever()
        self.ranker = self._build_ranker()
        self.kg_facts = self._build_kg()
        self.images = self._load_images()

        self.status["startup_seconds"] = round(time.perf_counter() - started, 1)
        log.info("Engine ready: %s", self.status)

    # ---- components ----

    def _build_llm(self):
        if self.settings.llm == "offline":
            from llm.offline_llm import OfflineLLM

            self.status["llm"] = "offline (rule-based demo extractor, not for evaluation)"
            log.warning("GC_LLM=offline: using the rule-based demo extractor.")
            return OfflineLLM()

        from llm.llm_client import LLMClient, LLMConfigError

        try:
            client = LLMClient.from_env()
        except LLMConfigError as exc:
            raise EngineUnavailable(
                f"{exc} Fill in FOUNDRY_* in .env, or set GC_LLM=offline to try the demo extractor."
            ) from exc
        self.status["llm"] = f"foundry ({client.deployment})"
        return client

    def _build_retriever(self):
        mode = self.settings.retriever
        index_dir = self.settings.semantic_index

        if mode in {"auto", "semantic"}:
            try:
                from retrieval.config import INDEX_FILE
                from retrieval.semantic_search import SemanticRetriever

                if not (index_dir / INDEX_FILE).exists():
                    raise FileNotFoundError(
                        f"No FAISS index in {index_dir}. Run CODE/retrieval/embed_corpus.py "
                        "and build_faiss_index.py.")
                retriever = SemanticRetriever(index_dir)
                self.status["retriever"] = f"semantic ({retriever.manifest.get('model', 'BGE')}, {retriever.size} vectors)"
                return retriever
            except Exception as exc:   # missing faiss / torch / index
                if mode == "semantic":
                    raise EngineUnavailable(f"Semantic retrieval unavailable: {exc}") from exc
                log.warning("Semantic retrieval unavailable (%s); using the TF-IDF stand-in.", exc)
                self.status["retriever_warning"] = str(exc)

        self.status["retriever"] = "tfidf stand-in (dev only: semantic index or packages missing)"
        return TfidfStandInRetriever(self.products)

    def _build_ranker(self):
        from llm import stub_ranker

        index_dir = self.settings.rank_index
        if not index_dir.exists():
            self.status["ranker"] = f"stub ranker (no BM25 index at {index_dir})"
            return stub_ranker.rank
        try:
            from CODE.ranking.pipeline import GiftRanker
            from llm.ranking_adapter import HybridRanker

            ranker = HybridRanker(GiftRanker(str(index_dir), load_tfidf=False))
        except Exception as exc:   # bm25s missing, unreadable index, ...
            log.warning("GiftRanker unavailable (%s); using the stub ranker.", exc)
            self.status["ranker"] = f"stub ranker (GiftRanker failed: {exc})"
            return stub_ranker.rank
        self.status["ranker"] = f"GiftRanker: BM25 + RRF + filters + ranking + MMR ({index_dir})"
        return ranker

    def _build_kg(self):
        mode = self.settings.kg
        if mode == "off":
            self.status["kg"] = "off"
            return None
        try:
            from knowledge_graph.api import load_kg
            from llm.kg_adapter import KGEvidence

            if self.settings.kg_export:
                kg = load_kg(export_dir=str(self.settings.kg_export))
            else:
                kg = load_kg(str(self.settings.corpus))
        except Exception as exc:
            if mode == "on":
                raise EngineUnavailable(f"Knowledge graph unavailable: {exc}") from exc
            log.warning("Knowledge graph unavailable (%s); explanations use product data only.", exc)
            self.status["kg"] = f"off ({exc})"
            return None
        self.status["kg"] = "on"
        return KGEvidence(kg)

    def _load_images(self):
        import json

        path = self.settings.image_lookup
        if not path:
            self.status["images"] = "none (set GC_IMAGE_LOOKUP to the build_image_lookup.py output)"
            return {}
        try:
            with open(path, encoding="utf-8") as infile:
                images = json.load(infile)
        except (OSError, ValueError) as exc:
            log.warning("Could not read image lookup %s: %s", path, exc)
            self.status["images"] = f"none ({exc})"
            return {}
        self.status["images"] = f"{len(images)} from {path}"
        return images

    # ---- per conversation ----

    def new_conversation(self):
        extra = {"ranker": self.ranker}
        if self.kg_facts is not None:
            extra["kg_facts"] = self.kg_facts
        return self._GiftConcierge(self.llm, self.retriever, self.products,
                                   top_k=self.settings.top_k,
                                   log_path=REPO_ROOT / "logs" / "state_changes.jsonl",
                                   **extra)

    def image_for(self, parent_asin):
        """Image URL for a product from the build_image_lookup.py output, or None.
        Accepts plain URL values or an Amazon `images` entry ({hi_res, large, thumb})."""
        value = self.images.get(parent_asin)
        if isinstance(value, dict):
            value = value.get("hi_res") or value.get("large") or value.get("thumb")
        elif isinstance(value, list) and value:
            first = value[0]
            value = first.get("hi_res") or first.get("large") or first.get("thumb") if isinstance(first, dict) else first
        return value if isinstance(value, str) and value.startswith("http") else None
