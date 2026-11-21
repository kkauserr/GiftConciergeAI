"""Public integration API of the Gift Concierge knowledge graph.

This is the only module other components need. It is independent of any LLM
provider and of the NetworkX internals: callers get plain JSON-serializable
dicts described in INTEGRATION.md (contract version CONTRACT_VERSION).

    from knowledge_graph.api import load_kg

    kg = load_kg()                       # build or load once (cached)
    resp = kg.get_evidence(["B0164RRP14", "1683239822"],
                           query_interests=["fishing_hunting"], max_reviews=3)

A module-level get_evidence() using the cached default graph is also provided.

The API only *reads* what inference already stored in the graph; it never
re-derives interests, aspects or review polarity. In particular it never
turns negative or mixed review evidence into a positive claim: review
qualities come only from review-aggregate HAS_ASPECT edges (positive by
construction and checked by KnowledgeGraph.validate()), concerns come only
from negative-majority signals, and individual observations keep their own
polarity.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .graph import KnowledgeGraph, iter_jsonl
from .ontology import (
    ASPECT_ATTRIBUTE, ASPECT_QUALITY, EVIDENCE_STRUCTURED, NodeType, RelType, aspect_id,
    interest_id, product_id,
)

CONTRACT_VERSION = "kg-evidence/1.0"
DEFAULT_SAMPLE = Path(__file__).resolve().parents[2] / "DATA" / "samples" / "sample_corpus_v2.jsonl"

DEFAULT_MAX_REVIEWS = 3
MAX_REVIEWS_LIMIT = 10        # hard cap on max_reviews
MAX_SNIPPET_CHARS = 300       # review snippets longer than this are truncated
MAX_ITEM_EVIDENCE = 3         # listing evidence items shown per interest/attribute

_LABEL_NORMALIZE = re.compile(r"[\s\-]+")


def _snip(text: Optional[str], limit: int = MAX_SNIPPET_CHARS) -> Optional[str]:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _prov(item: Dict[str, Any]) -> Dict[str, Any]:
    """Public view of one stored evidence item."""
    out = {
        "source": item.get("source"),
        "fields": list(item.get("source_fields") or []),
        "text": _snip(item.get("evidence")),
        "confidence": item.get("confidence"),
        "rule_id": item.get("rule_id"),
    }
    if item.get("context") and item.get("context") != item.get("evidence"):
        out["context"] = _snip(item.get("context"))
    return out


class KGService:
    """A loaded, read-only knowledge graph plus the get_evidence() API.

    Build it once (from_jsonl / from_export / from_graph) and reuse it for
    every request; get_evidence() never rebuilds or mutates the graph.
    Safe to share across threads for reads.
    """

    def __init__(self, kg: KnowledgeGraph, source: Optional[str] = None):
        self.kg = kg
        self.source = source
        self._interests = {
            kg.g.nodes[n]["label"]: kg.g.nodes[n] for n in kg.nodes_of_type(NodeType.INTEREST)}
        self._aspects = {
            kg.g.nodes[n]["label"]: kg.g.nodes[n] for n in kg.nodes_of_type(NodeType.ASPECT)}
        if not self._interests:
            raise ValueError("graph has no interest vocabulary; build it with inference enabled")

    # ------------------------------------------------------------ loading
    @classmethod
    def from_jsonl(cls, path: str | Path = DEFAULT_SAMPLE, limit: Optional[int] = None) -> "KGService":
        """Build the graph (all inference) from a corpus JSONL file."""
        return cls(KnowledgeGraph.build(iter_jsonl(path, limit)), str(path))

    @classmethod
    def from_export(cls, out_dir: str | Path) -> "KGService":
        """Load a graph previously written with `build_kg --out DIR`. Use
        this for large graphs: nothing is re-extracted or re-inferred."""
        return cls(KnowledgeGraph.from_export(out_dir), str(out_dir))

    @classmethod
    def from_graph(cls, kg: KnowledgeGraph) -> "KGService":
        return cls(kg, "in-memory")

    # ------------------------------------------------------- vocabulary
    @property
    def vocab_versions(self) -> Dict[str, str]:
        return dict(self.kg.vocab_versions)

    def interest_labels(self) -> List[str]:
        """The exact labels accepted in query_interests."""
        return sorted(self._interests)

    def interest_vocabulary(self) -> List[Dict[str, Any]]:
        """Labels with display names, descriptions and example phrasings
        (aliases) for building the LLM's extraction prompt/schema."""
        return [{"label": k, "display_name": v.get("display_name"),
                 "description": v.get("description"), "aliases": list(v.get("aliases") or [])}
                for k, v in sorted(self._interests.items())]

    def info(self) -> Dict[str, Any]:
        stats = self.kg.stats()
        return {"contract_version": CONTRACT_VERSION, "source": self.source,
                "vocabulary": self.vocab_versions, "nodes": stats["nodes"], "edges": stats["edges"]}

    # ------------------------------------------------------- main API
    def get_evidence(self, parent_asins: Sequence[str], query_interests: Optional[Iterable[str]] = None,
                     max_reviews: int = DEFAULT_MAX_REVIEWS) -> Dict[str, Any]:
        """Structured evidence for each product, in input order.

        parent_asins     ordered list of parent_asin strings (duplicates are
                         answered in place; unknown ones get found=false).
        query_interests  optional interest labels from interests.json. Unknown
                         labels are ignored with a warning; they never fail
                         the call and never filter products.
        max_reviews      max individual reviews quoted per aspect (0..10).
        """
        if isinstance(parent_asins, (str, bytes)) or not isinstance(parent_asins, Sequence):
            raise TypeError("parent_asins must be a list of parent_asin strings")
        if isinstance(max_reviews, bool) or not isinstance(max_reviews, int) or max_reviews < 0:
            raise ValueError("max_reviews must be a non-negative integer")
        warnings: List[Dict[str, Any]] = []
        if max_reviews > MAX_REVIEWS_LIMIT:
            warnings.append({"code": "max_reviews_capped", "value": max_reviews,
                             "message": f"max_reviews capped at {MAX_REVIEWS_LIMIT}"})
            max_reviews = MAX_REVIEWS_LIMIT
        accepted = self._check_interests(query_interests, warnings)

        results = []
        for i, asin in enumerate(parent_asins):
            if not isinstance(asin, str) or not asin.strip():
                warnings.append({"code": "invalid_asin", "value": repr(asin), "index": i,
                                 "message": "parent_asin must be a non-empty string"})
                results.append(self._missing(asin if isinstance(asin, str) else None, i))
                continue
            asin = asin.strip()
            if product_id(asin) not in self.kg.g:
                warnings.append({"code": "unknown_asin", "value": asin, "index": i,
                                 "message": "parent_asin is not in the knowledge graph"})
                results.append(self._missing(asin, i))
                continue
            results.append(self._product(asin, i, accepted, max_reviews))
        return {
            "contract_version": CONTRACT_VERSION,
            "vocabulary": self.vocab_versions,
            "query": {"parent_asins": list(parent_asins) if all(isinstance(a, str) for a in parent_asins)
                      else [a if isinstance(a, str) else repr(a) for a in parent_asins],
                      "query_interests": accepted, "max_reviews": max_reviews},
            "warnings": warnings,
            "results": results,
        }

    # ------------------------------------------------------- internals
    def _check_interests(self, query_interests, warnings) -> List[str]:
        if query_interests is None:
            return []
        if isinstance(query_interests, (str, bytes)):
            raise TypeError("query_interests must be a list of labels, not a single string")
        accepted: List[str] = []
        for raw in query_interests:
            if not isinstance(raw, str):
                warnings.append({"code": "unknown_interest", "value": repr(raw),
                                 "message": "interest labels must be strings"})
                continue
            label = _LABEL_NORMALIZE.sub("_", raw.strip().lower())
            if label not in self._interests:
                warnings.append({"code": "unknown_interest", "value": raw,
                                 "message": "not a label in interests.json; ignored"})
                continue
            if label != raw:
                warnings.append({"code": "normalized_interest", "value": raw, "normalized": label})
            if label not in accepted:
                accepted.append(label)
        return accepted

    @staticmethod
    def _missing(asin: Optional[str], index: int) -> Dict[str, Any]:
        return {"index": index, "parent_asin": asin, "found": False, "product": None,
                "brand": None, "category": None, "interests": [],
                "query_interest_match": None, "attributes": [], "review_qualities": [],
                "review_concerns": [], "review_observations": [], "review_summary": None,
                "grounding": {"assertable_claims": [], "cautions": []}, "reasoning_paths": []}

    def _product(self, asin: str, index: int, accepted: List[str], max_reviews: int) -> Dict[str, Any]:
        kg, pid = self.kg, product_id(asin)
        node = kg.g.nodes[pid]
        paths: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        cautions: List[Dict[str, Any]] = []

        def add_path(claim, kind, edges, assertable):
            pid_ = f"p{len(paths) + 1}"
            paths.append({"id": pid_, "kind": kind, "claim": claim, "assertable": assertable,
                          "edges": [{"src": s, "rel": r, "dst": d} for s, r, d in edges]})
            return pid_

        # identity, brand, category ------------------------------------
        product = {k: node.get(k) for k in (
            "title", "domain", "price_num", "average_rating", "rating_number", "store",
            "age_min", "age_max", "review_count", "average_review_rating")}
        product["price"] = product.pop("price_num")
        b = kg.brand_of(asin)
        brand = None
        if b:
            e = kg.edge(pid, RelType.MADE_BY, b["id"])
            brand = {"id": b["id"], "name": b["name"],
                     "kind": "author" if "author" in (b.get("kinds") or []) else "brand",
                     "provenance": _prov(e)}
            add_path(f"{'Written' if brand['kind'] == 'author' else 'Made'} by {b['name']}",
                     "brand", [(pid, "MADE_BY", b["id"])], True)
        path = kg.category_path_of(asin)
        category = None
        leaves = kg._out(pid, RelType.BELONGS_TO)
        if leaves:
            e = kg.edge(pid, RelType.BELONGS_TO, leaves[0])
            category = {"id": leaves[0], "path": path, "from_domain_fallback": e["source"] != "metadata",
                        "provenance": _prov(e)}

        # interests -----------------------------------------------------
        interests = []
        for it in kg.interests_of(asin):
            iid = interest_id(it["label"])
            e = kg.edge(pid, RelType.MATCHES, iid)
            display = self._interests[it["label"]].get("display_name") or it["label"]
            sources = {x["source"] for x in e["evidence_list"]}
            if it["match_type"] == "themed":
                claim = f"Has a {display} theme (stated in the listing)"
            elif "category_map" in sources:
                claim = f"Matches the interest {display} (product category)"
            else:
                claim = f"Matches the interest {display} (book title)"
            pid_ = add_path(claim, "interest", [(pid, "MATCHES", iid)], True)
            claims.append({"text": claim, "basis": f"interest_{it['match_type']}", "path_id": pid_})
            interests.append({
                "label": it["label"], "display_name": display, "match_type": it["match_type"],
                "confidence": e["confidence"], "matches_query": it["label"] in accepted,
                "evidence": [_prov(x) for x in sorted(e["evidence_list"], key=lambda x: -x["confidence"])
                             [:MAX_ITEM_EVIDENCE]],
                "path_id": pid_})
        matched = [i["label"] for i in interests if i["matches_query"]]
        query_match = None if not accepted else {
            "requested": accepted, "matched": matched,
            "unmatched": [l for l in accepted if l not in matched]}

        # objective attributes -------------------------------------------
        attributes = []
        for it in kg.aspects_of(asin, aspect_class=ASPECT_ATTRIBUTE):
            aid = aspect_id(it["label"])
            e = kg.edge(pid, RelType.HAS_ASPECT, aid)
            display = self._aspects[it["label"]].get("display_name") or it["label"]
            structured = any(x.get("evidence_type") == EVIDENCE_STRUCTURED for x in e["evidence_list"])
            basis = "structured_metadata" if structured else "seller_listing"
            noun = f"made of {display.lower()}" if it["label"].startswith("material_") else display.lower()
            claim = (f"{noun[0].upper() + noun[1:]} (product metadata)" if structured
                     else f"The listing says it is {noun}")
            pid_ = add_path(claim, "attribute", [(pid, "HAS_ASPECT", aid)], True)
            claims.append({"text": claim, "basis": basis, "path_id": pid_})
            attributes.append({
                "label": it["label"], "display_name": display, "basis": basis,
                "confidence": e["confidence"],
                "evidence": [_prov(x) for x in sorted(e["evidence_list"], key=lambda x: -x["confidence"])
                             [:MAX_ITEM_EVIDENCE]],
                "path_id": pid_})

        # review observations (individual reviews) ------------------------
        observations_by_aspect: Dict[str, List[Dict[str, Any]]] = {}
        for ev in kg.review_evidence(asin):
            rnode = kg.g.nodes[ev["review_id"]]
            observations_by_aspect.setdefault(ev["aspect"], []).append({
                "review_id": ev["review_id"], "aspect": ev["aspect"],
                "aspect_display_name": self._aspects[ev["aspect"]].get("display_name"),
                "polarity": ev["polarity"], "rating": ev["rating"],
                "verified_purchase": ev["verified_purchase"],
                "helpful_votes": rnode.get("helpful_vote"),
                "rating_conflict": bool(ev["rating_conflict"]),
                "snippets": [{"text": _snip(x["text"]), "polarity": x["polarity"],
                              "confidence": x["confidence"], "rule_id": x["rule_id"]}
                             for x in ev["evidence"]],
                "confidence": max(x["confidence"] for x in ev["evidence"]),
            })
        observations, truncated = [], {}
        for aspect in sorted(observations_by_aspect):
            obs = sorted(observations_by_aspect[aspect],
                         key=lambda o: (o["rating_conflict"], -o["confidence"],
                                        -(o["helpful_votes"] or 0), o["review_id"]))
            if len(obs) > max_reviews:
                truncated[aspect] = len(obs) - max_reviews
            for o in obs[:max_reviews]:
                o["path_id"] = add_path(
                    f"One review gives {o['polarity']} evidence about {o['aspect_display_name']}",
                    "review_observation",
                    [(o["review_id"], "ABOUT", pid), (o["review_id"], "SUPPORTS", aspect_id(aspect))],
                    False)
                observations.append(o)

        # aggregated review conclusions (positive only) -------------------
        qualities = []
        for ra in kg.review_aspects_of(asin):
            aid = aspect_id(ra["label"])
            e = kg.edge(pid, RelType.HAS_ASPECT, aid)
            display = self._aspects[ra["label"]].get("display_name") or ra["label"]
            supporting = [{"review_id": x["review_id"], "text": _snip(x["evidence"]),
                           "rating": kg.g.nodes[x["review_id"]].get("rating"),
                           "verified_purchase": kg.g.nodes[x["review_id"]].get("verified_purchase")}
                          for x in e["evidence_list"]]
            claim = f"{ra['n_positive']} independent reviews praise its {display.lower()}"
            edges = [(pid, "HAS_ASPECT", aid)]
            for s in supporting[:max_reviews]:
                edges += [(s["review_id"], "ABOUT", pid), (s["review_id"], "SUPPORTS", aid)]
            pid_ = add_path(claim, "review_quality", edges, True)
            claims.append({"text": claim, "basis": "review_aggregate", "path_id": pid_})
            if ra["n_negative"] or ra["n_mixed"]:
                cautions.append({"text": f"{ra['n_negative'] + ra['n_mixed']} review(s) disagree about "
                                         f"{display.lower()}", "path_id": pid_})
            qualities.append({
                "label": ra["label"], "display_name": display, "conclusion": "supported",
                "polarity": "positive", "confidence": e["confidence"],
                "counts": {"positive": e["n_positive"], "negative": e["n_negative"],
                           "mixed": e["n_mixed"], "neutral": e["n_neutral"],
                           "rating_conflict": e["n_rating_conflict"],
                           "positive_verified": e["n_positive_verified"],
                           "reviews_in_graph": e["n_reviews"]},
                "supporting_reviews": supporting[:max_reviews],
                "supporting_reviews_omitted": max(0, len(supporting) - max_reviews),
                "path_id": pid_})

        # negative concerns ------------------------------------------------
        concerns = []
        for label, sig in sorted(kg.review_concerns_of(asin).items()):
            aid = aspect_id(label)
            display = self._aspects[label].get("display_name") or label
            negs = [o for o in observations_by_aspect.get(label, []) if o["polarity"] == "negative"]
            negs = sorted(negs, key=lambda o: (-o["confidence"], o["review_id"]))
            claim = f"{sig['negative']} reviews report problems with {display.lower()}"
            edges = []
            for o in negs[:max_reviews]:
                edges += [(o["review_id"], "ABOUT", pid), (o["review_id"], "SUPPORTS", aid)]
            pid_ = add_path(claim, "review_concern", edges, False)
            cautions.append({"text": claim, "path_id": pid_})
            concerns.append({
                "label": label, "display_name": display, "status": "concern", "polarity": "negative",
                "counts": {k: sig[k] for k in ("positive", "negative", "mixed", "neutral", "rating_conflict")},
                "examples": [{"review_id": o["review_id"], "rating": o["rating"],
                              "verified_purchase": o["verified_purchase"],
                              "text": _snip(" / ".join(s["text"] for s in o["snippets"]
                                                       if s["polarity"] == "negative"))}
                             for o in negs[:max_reviews]],
                "path_id": pid_})

        rids = kg._in(pid, RelType.ABOUT)
        signals = node.get("review_aspect_signals") or {}
        review_summary = {
            "reviews_in_graph": len(rids),
            "verified_reviews_in_graph": sum(bool(kg.g.nodes[r].get("verified_purchase")) for r in rids),
            "review_count_in_corpus": node.get("review_count"),
            "aspects_with_observations": sorted(observations_by_aspect),
            "observations_omitted": truncated,
            "aspect_status": {k: v.get("status") for k, v in sorted(signals.items())},
        }
        return {
            "index": index, "parent_asin": asin, "found": True, "product": product,
            "brand": brand, "category": category, "interests": interests,
            "query_interest_match": query_match, "attributes": attributes,
            "review_qualities": qualities, "review_concerns": concerns,
            "review_observations": observations, "review_summary": review_summary,
            "grounding": {"assertable_claims": claims, "cautions": cautions},
            "reasoning_paths": paths,
        }


def evidence_to_text(result: Dict[str, Any], max_quotes: int = 2) -> str:
    """Compact, provider-neutral text for one product result, for use as
    grounding context in an LLM prompt. Lists only assertable claims, the
    cautions (concerns / disagreement) and short verbatim review quotes,
    each tagged with its reasoning-path id so the model can cite it."""
    if not result.get("found"):
        return f"[{result.get('parent_asin')}] not in the knowledge graph: no evidence available."
    p, lines = result["product"], []
    who = result["brand"]
    lines.append(f"[{result['parent_asin']}] {p['title']}")
    meta = [p["domain"]]
    if who:
        meta.append(f"{'by' if who['kind'] == 'author' else 'brand'} {who['name']}")
    if result["category"]:
        meta.append("category " + " > ".join(result["category"]["path"]))
    lines.append("  " + "; ".join(m for m in meta if m))
    for c in result["grounding"]["assertable_claims"]:
        lines.append(f"  + {c['text']} [{c['path_id']}]")
    for q in result["review_qualities"]:
        for s in q["supporting_reviews"][:max_quotes]:
            lines.append(f"    \"{s['text']}\" ({q['display_name']}, review rated {s['rating']})")
    for c in result["grounding"]["cautions"]:
        lines.append(f"  ! {c['text']} [{c['path_id']}]")
    for c in result["review_concerns"]:
        for ex in c["examples"][:max_quotes]:
            lines.append(f"    \"{ex['text']}\" ({c['display_name']}, review rated {ex['rating']})")
    if not result["grounding"]["assertable_claims"] and not result["grounding"]["cautions"]:
        lines.append("  (no inferred interests, attributes or review conclusions)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default, process-wide graph
# ---------------------------------------------------------------------------

_CACHE: Dict[str, KGService] = {}
_LOCK = threading.Lock()


def load_kg(path: str | Path | None = None, export_dir: str | Path | None = None,
            reload: bool = False) -> KGService:
    """Return a cached KGService. Builds from `path` (default: the
    900-product sample) or loads `export_dir` (a `build_kg --out` export)
    the first time; later calls with the same arguments reuse it."""
    if path is not None and export_dir is not None:
        raise ValueError("pass either path or export_dir, not both")
    key = f"export:{Path(export_dir).resolve()}" if export_dir is not None \
        else f"jsonl:{Path(path or DEFAULT_SAMPLE).resolve()}"
    with _LOCK:
        if reload or key not in _CACHE:
            _CACHE[key] = (KGService.from_export(export_dir) if export_dir is not None
                           else KGService.from_jsonl(path or DEFAULT_SAMPLE))
        return _CACHE[key]


def get_evidence(parent_asins: Sequence[str], query_interests: Optional[Iterable[str]] = None,
                 max_reviews: int = DEFAULT_MAX_REVIEWS) -> Dict[str, Any]:
    """get_evidence() on the default graph (see load_kg)."""
    return load_kg().get_evidence(parent_asins, query_interests, max_reviews)
