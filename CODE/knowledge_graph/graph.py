"""In-memory KG built on NetworkX (for the development sample).

Wraps a MultiDiGraph keyed by relationship type, so each (src, rel, dst)
exists at most once. Node attribute `type` holds the NodeType value. Edge
attributes hold the provenance fields.

Downstream code should use the query methods on KnowledgeGraph, not raw
NetworkX calls. The full-corpus backend can then be swapped in later
without changing the callers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import networkx as nx

from .extractors import DEFAULT_EXTRACTORS, Extraction, RecordError, extract_record
from .inference import make_aspect_extractor, make_interest_extractor
from .ontology import (
    ASPECT_ATTRIBUTE, ASPECT_QUALITY, EDGE_SCHEMA, EVIDENCE_LISTING_STATED,
    EVIDENCE_REVIEW, EVIDENCE_STRUCTURED, ITEM_POLARITIES, SOURCE_FEATURE_LEXICON,
    SOURCE_REVIEW_AGGREGATE, SOURCE_REVIEW_TEXT, Edge, Node, NodeType, Polarity, RelType,
    product_id,
)
from .review_inference import make_review_aspect_extractor, review_polarity_of_items
from .vocabulary import Vocabulary, load_aspects, load_interests

EXPORT_FORMAT = "gift-concierge-kg-export/1"

# Node attributes that are unioned across records, not first-wins.
_LIST_MERGE_ATTRS = {"kinds", "domains"}


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.flag_counts: Counter = Counter()
        self.skipped_records = 0
        self.vocab_versions: Dict[str, str] = {}
        self.aspect_classes: Dict[str, str] = {}  # aspect node id -> class
        self.features_corroborate_only = False     # interest vocabulary policy

    # ---------------------------------------------------------------- build
    def add_node(self, node: Node) -> None:
        if node.id not in self.g:
            self.g.add_node(node.id, type=node.type.value, **node.attrs)
            return
        existing = self.g.nodes[node.id]
        if node.type == NodeType.PRODUCT:
            # The same parent_asin appearing twice in the corpus. First record wins.
            self.flag_counts["duplicate_product_records"] += 1
        if existing["type"] != node.type.value:
            raise ValueError(f"node {node.id} type conflict: "
                             f"{existing['type']} vs {node.type.value}")
        for key, value in node.attrs.items():
            if key in _LIST_MERGE_ATTRS:
                merged = list(existing.get(key, []))
                merged.extend(v for v in value if v not in merged)
                existing[key] = merged
            elif key == "name" and value != existing.get("name"):
                # Same slug, different display form (e.g. "LEGO" / "Lego").
                aliases = existing.setdefault("aliases", [])
                if value not in aliases:
                    aliases.append(value)
            elif key == "duplicate_count":
                existing[key] = existing.get(key, 0) + value
            # Any other scalar: keep the first value seen.

    def add_edge(self, edge: Edge) -> None:
        """Add an edge, or merge its evidence into the existing (src, rel, dst).

        Each edge keeps `evidence_list`, one entry per distinct evidence item.
        Items are de-duplicated on (source, source_fields, evidence, span,
        rule_id), so re-adding identical evidence (e.g. a SUBCATEGORY_OF edge
        seen from many products) changes nothing. The top-level provenance
        fields always mirror the highest-confidence item, and `confidence` is
        the max over items (ontology.AGGREGATION_RULE). No combining that
        could inflate scores.
        """
        for end in (edge.src, edge.dst):
            if end not in self.g:
                raise ValueError(f"edge {edge.rel.value} references missing node {end}")
        key = edge.rel.value
        item = _evidence_item(edge)
        if self.g.has_edge(edge.src, edge.dst, key=key):
            data = self.g.edges[edge.src, edge.dst, key]
            if any(_evidence_key(e) == _evidence_key(item) for e in data["evidence_list"]):
                return
            data["evidence_list"].append(item)
            for k, v in edge.attrs.items():
                data.setdefault(k, v)
            best = max(data["evidence_list"], key=lambda e: e["confidence"])
            data.update(best)
            return
        d = edge.to_dict()
        for k in ("src", "dst", "rel"):
            d.pop(k)
        self.g.add_edge(edge.src, edge.dst, key=key, rel=key,
                        evidence_list=[item], **d)

    def add_extraction(self, ex: Extraction) -> None:
        for node in ex.nodes:
            self.add_node(node)
        for edge in ex.edges:
            self.add_edge(edge)
        for k, v in ex.flags.items():
            if isinstance(v, bool):
                if v:
                    self.flag_counts[k] += 1
            elif isinstance(v, int):
                self.flag_counts[k] += v
            else:
                self.flag_counts[f"{k}={v}"] += 1

    def add_vocabulary(self, vocab: Vocabulary) -> None:
        """Add every vocabulary concept as a node, matched or not, so the
        graph's label set always equals the shared vocabulary. That includes
        the review-only quality aspects (targets of SUPPORTS edges)."""
        for c in vocab.concepts + vocab.reserved:
            self.add_node(Node(c.node_id, c.node_type, c.node_attrs()))
        self.vocab_versions[vocab.name] = vocab.version_tag
        if vocab.node_type == NodeType.INTEREST:
            self.features_corroborate_only = vocab.features_corroborate_only
        self.aspect_classes.update(
            {c.node_id: c.aspect_class for c in vocab.concepts + vocab.reserved if c.aspect_class})

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]],
                     extractors: Optional[list] = None,
                     vocabularies: Optional[list] = None) -> "KnowledgeGraph":
        kg = cls()
        for vocab in vocabularies or []:
            kg.add_vocabulary(vocab)
        for record in records:
            try:
                kg.add_extraction(extract_record(record, extractors))
            except RecordError:
                kg.skipped_records += 1
        return kg

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: Optional[int] = None,
                   extractors: Optional[list] = None,
                   vocabularies: Optional[list] = None) -> "KnowledgeGraph":
        return cls.from_records(iter_jsonl(path, limit), extractors, vocabularies)

    @classmethod
    def build(cls, records: Iterable[Dict[str, Any]], inference: bool = True,
              interests: Optional[Vocabulary] = None,
              aspects: Optional[Vocabulary] = None,
              review_inference: bool = True,
              review_classifier=None) -> "KnowledgeGraph":
        """Direct extractors plus (by default) interest, attribute-aspect and
        review-evidence inference using the versioned vocabularies.
        `review_classifier` replaces the rule-based review classifier (any
        object with `version` and `classify(text, domain)`)."""
        if not inference:
            return cls.from_records(records)
        interests = interests or load_interests()
        aspects = aspects or load_aspects()
        extractors = DEFAULT_EXTRACTORS + [make_interest_extractor(interests),
                                           make_aspect_extractor(aspects)]
        if review_inference:
            extractors.append(make_review_aspect_extractor(aspects, review_classifier))
        return cls.from_records(records, extractors, [interests, aspects])

    # -------------------------------------------------------------- queries
    def node(self, node_id: str) -> Dict[str, Any]:
        return dict(self.g.nodes[node_id], id=node_id)

    def nodes_of_type(self, node_type: NodeType) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d["type"] == node_type.value]

    def _out(self, node_id: str, rel: RelType) -> List[str]:
        return [v for _, v, k in self.g.out_edges(node_id, keys=True) if k == rel.value]

    def _in(self, node_id: str, rel: RelType) -> List[str]:
        return [u for u, _, k in self.g.in_edges(node_id, keys=True) if k == rel.value]

    def edge(self, src: str, rel: RelType, dst: str) -> Dict[str, Any]:
        return dict(self.g.edges[src, dst, rel.value])

    def brand_of(self, parent_asin: str) -> Optional[Dict[str, Any]]:
        out = self._out(product_id(parent_asin), RelType.MADE_BY)
        return self.node(out[0]) if out else None

    def category_path_of(self, parent_asin: str) -> List[str]:
        """Root-to-leaf category names, found by walking SUBCATEGORY_OF."""
        leaves = self._out(product_id(parent_asin), RelType.BELONGS_TO)
        if not leaves:
            return []
        chain, cur = [], leaves[0]
        while cur:
            chain.append(self.g.nodes[cur]["name"])
            parents = self._out(cur, RelType.SUBCATEGORY_OF)
            cur = parents[0] if parents else None
        return chain[::-1]

    def subcategories(self, category: str, recursive: bool = True) -> List[str]:
        direct = self._in(category, RelType.SUBCATEGORY_OF)
        if not recursive:
            return direct
        out, stack = [], list(direct)
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(self._in(c, RelType.SUBCATEGORY_OF))
        return out

    def products_in_category(self, category: str, recursive: bool = True) -> List[str]:
        cats = [category] + (self.subcategories(category) if recursive else [])
        return sorted({p for c in cats for p in self._in(c, RelType.BELONGS_TO)})

    def products_by_brand(self, brand: str) -> List[str]:
        return sorted(self._in(brand, RelType.MADE_BY))

    def reviews_of(self, parent_asin: str) -> List[Dict[str, Any]]:
        return [self.node(r) for r in self._in(product_id(parent_asin), RelType.ABOUT)]

    def _concepts_of(self, parent_asin: str, rel: RelType,
                     min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        pid = product_id(parent_asin)
        out = []
        for c in self._out(pid, rel):
            e = self.edge(pid, rel, c)
            if e["confidence"] < min_confidence:
                continue
            node = self.g.nodes[c]
            item = {"label": node["label"], "confidence": e["confidence"],
                    "sources": sorted({x["source"] for x in e["evidence_list"]}),
                    "evidence": [{k: x[k] for k in ("source_fields", "evidence", "context",
                                                     "confidence", "rule_id") if x.get(k) is not None}
                                 for x in e["evidence_list"]]}
            if rel == RelType.HAS_ASPECT:
                item["evidence_types"] = sorted({x["evidence_type"] for x in e["evidence_list"]})
            out.append(item)
        return sorted(out, key=lambda d: (-d["confidence"], d["label"]))

    def interests_of(self, parent_asin: str, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Matched interests with evidence. Empty is a normal result: MATCHES
        is optional and must never be used to exclude a product."""
        items = self._concepts_of(parent_asin, RelType.MATCHES, min_confidence)
        pid = product_id(parent_asin)
        for it in items:
            it["match_type"] = self.edge(pid, RelType.MATCHES, f"interest:{it['label']}").get("match_type")
        return items

    def aspects_of(self, parent_asin: str, min_confidence: float = 0.0,
                   aspect_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """HAS_ASPECT targets with evidence. aspect_class="attribute" gives
        listing/metadata facts, "quality" gives review-supported qualities."""
        items = self._concepts_of(parent_asin, RelType.HAS_ASPECT, min_confidence)
        if aspect_class is None:
            return items
        return [it for it in items if self.aspect_classes.get(f"aspect:{it['label']}") == aspect_class]

    def review_aspects_of(self, parent_asin: str) -> List[Dict[str, Any]]:
        """Review-supported quality aspects with their independent-review counts."""
        pid = product_id(parent_asin)
        out = []
        for it in self.aspects_of(parent_asin, aspect_class=ASPECT_QUALITY):
            e = self.edge(pid, RelType.HAS_ASPECT, f"aspect:{it['label']}")
            out.append({"label": it["label"], "confidence": e["confidence"],
                        "n_positive": e["n_positive"], "n_negative": e["n_negative"],
                        "n_mixed": e["n_mixed"], "n_reviews": e["n_reviews"],
                        "n_positive_verified": e["n_positive_verified"],
                        "evidence": [{"review_id": x["review_id"], "evidence": x["evidence"]}
                                     for x in e["evidence_list"]]})
        return out

    def review_evidence(self, parent_asin: str, aspect_label: Optional[str] = None,
                        polarity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Per-review SUPPORTS evidence for a product, optionally filtered by
        aspect and by review-level polarity (positive|negative|neutral|mixed).
        Negative and mixed evidence is returned as such, never as support."""
        out = []
        for rid in self._in(product_id(parent_asin), RelType.ABOUT):
            for _, aid, k, d in self.g.out_edges(rid, keys=True, data=True):
                if k != RelType.SUPPORTS.value:
                    continue
                label = self.g.nodes[aid]["label"]
                if aspect_label and label != aspect_label:
                    continue
                if polarity and d["polarity"] != polarity:
                    continue
                out.append({"review_id": rid, "aspect": label, "polarity": d["polarity"],
                            "rating": d.get("rating"), "verified_purchase": d.get("verified_purchase"),
                            "rating_conflict": d.get("rating_conflict"),
                            "evidence": [{"text": x["evidence"], "polarity": x["evidence_polarity"],
                                          "confidence": x["confidence"], "rule_id": x["rule_id"]}
                                         for x in d["evidence_list"]]})
        return sorted(out, key=lambda r: (r["aspect"], r["polarity"], r["review_id"]))

    def review_concerns_of(self, parent_asin: str) -> Dict[str, Dict[str, Any]]:
        """Quality aspects where independent reviews are mostly negative.
        These are cautions for the conversation layer, never HAS_ASPECT."""
        signals = self.g.nodes[product_id(parent_asin)].get("review_aspect_signals") or {}
        return {k: v for k, v in signals.items() if v.get("status") == "concern"}

    def products_matching(self, interest_label: str, min_confidence: float = 0.0) -> Dict[str, float]:
        """parent_asin -> confidence for one interest, e.g. for a ranking boost."""
        iid = f"interest:{interest_label}"
        if iid not in self.g:
            return {}
        return {self.g.nodes[u]["parent_asin"]: self.edge(u, RelType.MATCHES, iid)["confidence"]
                for u in self._in(iid, RelType.MATCHES)
                if self.edge(u, RelType.MATCHES, iid)["confidence"] >= min_confidence}

    def product_summary(self, parent_asin: str) -> Dict[str, Any]:
        """Metadata facts plus inferred interests and aspects, with evidence,
        for one product. The later get_evidence() API will build on this."""
        pid = product_id(parent_asin)
        brand = self.brand_of(parent_asin)
        leaf = self._out(pid, RelType.BELONGS_TO)
        return {
            "parent_asin": parent_asin,
            "domain": self.g.nodes[pid].get("domain"),
            "title": self.g.nodes[pid].get("title"),
            "brand": brand and {"id": brand["id"], "name": brand["name"], "kinds": brand["kinds"]},
            "brand_provenance": brand and self.edge(pid, RelType.MADE_BY, brand["id"]),
            "category_path": self.category_path_of(parent_asin),
            "category_provenance": leaf and self.edge(pid, RelType.BELONGS_TO, leaf[0]),
            "interests": self.interests_of(parent_asin),
            "aspects": self.aspects_of(parent_asin, aspect_class=ASPECT_ATTRIBUTE),
            "reviews": len(self._in(pid, RelType.ABOUT)),
            "review_aspects": self.review_aspects_of(parent_asin),
            "review_concerns": self.review_concerns_of(parent_asin),
        }

    # ---------------------------------------------------------- validation
    def validate(self) -> List[str]:
        """Return a list of schema violations (empty when valid)."""
        problems = self._validate_structure()
        problems += self._validate_inferred()
        problems += self._validate_reviews()
        return problems

    def _validate_reviews(self) -> List[str]:
        """Rules for review evidence:
          SUPPORTS (Review -> quality Aspect)
            - target is a quality aspect; the review is ABOUT exactly one product
            - edge polarity is positive|negative|neutral|mixed and equals the
              polarity recomputed from its evidence items
            - every item is review_text evidence from this review, with an
              item polarity, and its span indexes this review's text inside
              the quoted clause (no evidence without review text)
          HAS_ASPECT to a quality aspect
            - polarity is positive and every item is positive
            - each item cites a distinct review that is ABOUT this product and
              has a positive, non-rating-conflict SUPPORTS edge to the aspect
            - the counts satisfy the stored aggregation policy
        """
        problems = []
        for u, v, k, d in self.g.edges(keys=True, data=True):
            if k == RelType.SUPPORTS.value:
                problems += self._check_supports(u, v, d)
            elif k == RelType.HAS_ASPECT.value and self.aspect_classes.get(v) == ASPECT_QUALITY:
                problems += self._check_review_has_aspect(u, v, d)
        return problems

    def _check_supports(self, rid: str, aid: str, d: Dict[str, Any]) -> List[str]:
        tag = f"SUPPORTS: {rid} -> {aid}"
        out = []
        if self.aspect_classes.get(aid) != ASPECT_QUALITY:
            out.append(f"{tag} targets a non-quality aspect")
        if len(self._out(rid, RelType.ABOUT)) != 1:
            out.append(f"{tag} review is not ABOUT exactly one product")
        items = d.get("evidence_list") or []
        if not items:
            return out + [f"{tag} has no evidence"]
        if d.get("polarity") not in {p.value for p in Polarity}:
            out.append(f"{tag} has invalid polarity {d.get('polarity')!r}")
        elif d["polarity"] != review_polarity_of_items(x.get("evidence_polarity") for x in items):
            out.append(f"{tag} polarity does not match its evidence items")
        text = self.g.nodes[rid].get("text") or ""
        for x in items:
            if x.get("source") != SOURCE_REVIEW_TEXT or x.get("evidence_type") != EVIDENCE_REVIEW:
                out.append(f"{tag} item is not review_text evidence")
            if x.get("review_id") != rid:
                out.append(f"{tag} item cites another review")
            if x.get("evidence_polarity") not in ITEM_POLARITIES:
                out.append(f"{tag} item has no valid evidence_polarity")
            span = x.get("span")
            if not span or not (0 <= span[0] < span[1] <= len(text)) or \
                    text[span[0]:span[1]] not in (x.get("evidence") or ""):
                out.append(f"{tag} item span does not point into the review text")
        return out

    def _check_review_has_aspect(self, pid: str, aid: str, d: Dict[str, Any]) -> List[str]:
        tag = f"HAS_ASPECT(review): {pid} -> {aid}"
        out = []
        items = d.get("evidence_list") or []
        if d.get("polarity") != Polarity.POSITIVE.value or \
                any(x.get("evidence_polarity") != Polarity.POSITIVE.value for x in items):
            out.append(f"{tag} is not purely positive")
        rids = [x.get("review_id") for x in items]
        if len(set(rids)) != len(rids):
            out.append(f"{tag} cites a review twice")
        for rid in rids:
            if rid not in self.g or pid not in self._out(rid, RelType.ABOUT):
                out.append(f"{tag} cites {rid}, which is not a review of this product")
                continue
            if not self.g.has_edge(rid, aid, key=RelType.SUPPORTS.value):
                out.append(f"{tag} cites {rid} without a SUPPORTS edge")
                continue
            s = self.g.edges[rid, aid, RelType.SUPPORTS.value]
            if s.get("polarity") != Polarity.POSITIVE.value or s.get("rating_conflict"):
                out.append(f"{tag} cites {rid}, whose evidence is not clean positive support")
        policy = d.get("aggregation_policy") or {}
        n_pos, crit = d.get("n_positive", 0), d.get("n_negative", 0) + d.get("n_mixed", 0)
        if n_pos != len(set(rids)) or n_pos < policy.get("min_positive_reviews", 2) or \
                n_pos < policy.get("positive_to_critical_ratio", 2.0) * crit:
            out.append(f"{tag} does not satisfy its aggregation policy")
        return out

    def _validate_inferred(self) -> List[str]:
        """Rules for MATCHES / HAS_ASPECT:
          - every edge has rule_id, vocab_version and a non-empty evidence_list
          - confidence equals the max over the evidence list, and each item is in [0, 1]
          - Books MATCHES never relies on feature text
          - MATCHES has match_type primary|themed and, when the interest
            vocabulary says features only corroborate, is never supported
            by feature text alone
          - HAS_ASPECT to a quality aspect requires review_aggregate evidence
          - HAS_ASPECT to an attribute aspect uses structured or listing_stated evidence
          - Interest and Aspect nodes come from a loaded vocabulary
        """
        problems = []
        for t in (NodeType.INTEREST, NodeType.ASPECT):
            for n in self.nodes_of_type(t):
                if "vocab_version" not in self.g.nodes[n]:
                    problems.append(f"{n} is not from a vocabulary")
        for u, v, k, d in self.g.edges(keys=True, data=True):
            if k not in (RelType.MATCHES.value, RelType.HAS_ASPECT.value):
                continue
            tag = f"{k}: {u} -> {v}"
            items = d.get("evidence_list") or []
            if not items or not d.get("rule_id") or not d.get("vocab_version"):
                problems.append(f"{tag} lacks evidence_list/rule_id/vocab_version")
                continue
            if any(not 0.0 <= x["confidence"] <= 1.0 for x in items) or \
                    d["confidence"] != max(x["confidence"] for x in items):
                problems.append(f"{tag} confidence is not the max of its evidence")
            if k == RelType.MATCHES.value and self.g.nodes[u].get("domain") == "Books" and \
                    any(x["source"] == SOURCE_FEATURE_LEXICON for x in items):
                problems.append(f"{tag} uses Book feature text")
            if k == RelType.MATCHES.value:
                if d.get("match_type") not in ("primary", "themed"):
                    problems.append(f"{tag} has no valid match_type")
                if self.features_corroborate_only and \
                        all(x["source"] == SOURCE_FEATURE_LEXICON for x in items):
                    problems.append(f"{tag} is supported by feature text only")
            if k == RelType.HAS_ASPECT.value:
                cls = self.g.nodes[v].get("aspect_class") or self.aspect_classes.get(v)
                types = {x.get("evidence_type") for x in items}
                if cls == ASPECT_QUALITY and (
                        types != {EVIDENCE_REVIEW} or
                        any(x["source"] != SOURCE_REVIEW_AGGREGATE for x in items)):
                    problems.append(f"{tag} quality aspect without review evidence")
                elif cls == ASPECT_ATTRIBUTE and not types <= {EVIDENCE_STRUCTURED, EVIDENCE_LISTING_STATED}:
                    problems.append(f"{tag} attribute aspect with evidence types {types}")
                elif cls is None:
                    problems.append(f"{tag} aspect has no aspect_class")
        return problems

    def _validate_structure(self) -> List[str]:
        problems = []
        for u, v, k in self.g.edges(keys=True):
            src_t, dst_t = EDGE_SCHEMA[RelType(k)]
            if self.g.nodes[u]["type"] != src_t.value or self.g.nodes[v]["type"] != dst_t.value:
                problems.append(f"{k}: {u} -> {v} violates {src_t.value}->{dst_t.value}")
        for p in self.nodes_of_type(NodeType.PRODUCT):
            for rel in (RelType.MADE_BY, RelType.BELONGS_TO):
                if len(self._out(p, rel)) > 1:
                    problems.append(f"{p} has multiple {rel.value} edges")
        cat = self.g.subgraph(self.nodes_of_type(NodeType.CATEGORY))
        if not nx.is_directed_acyclic_graph(nx.DiGraph(cat)):
            problems.append("SUBCATEGORY_OF contains a cycle")
        for c in cat.nodes:
            if len(self._out(c, RelType.SUBCATEGORY_OF)) > 1:
                problems.append(f"{c} has multiple parents")
        return problems

    # ---------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        node_counts = Counter(d["type"] for _, d in self.g.nodes(data=True))
        edge_counts = Counter(k for _, _, k in self.g.edges(keys=True))
        return {
            "nodes": dict(sorted(node_counts.items())),
            "edges": dict(sorted(edge_counts.items())),
            "total_nodes": self.g.number_of_nodes(),
            "total_edges": self.g.number_of_edges(),
            "skipped_records": self.skipped_records,
            "flags": dict(sorted(self.flag_counts.items())),
        }

    # --------------------------------------------------------------- export
    def export_jsonl(self, out_dir: str | Path) -> Dict[str, Path]:
        """Write nodes.jsonl, edges.jsonl and meta.json. The node/edge files
        are flat records that can later be converted to Neo4j bulk-import CSV
        or Parquet; from_export() loads all three back without re-running
        inference (useful for a graph built once from the full corpus)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes_path, edges_path = out / "nodes.jsonl", out / "edges.jsonl"
        meta_path = out / "meta.json"
        with open(nodes_path, "w", encoding="utf-8") as f:
            for n, d in self.g.nodes(data=True):
                f.write(json.dumps({"id": n, **d}, ensure_ascii=False) + "\n")
        with open(edges_path, "w", encoding="utf-8") as f:
            for u, v, d in self.g.edges(data=True):
                f.write(json.dumps({"src": u, "dst": v, **d}, ensure_ascii=False) + "\n")
        meta = {"format": EXPORT_FORMAT, "vocab_versions": self.vocab_versions,
                "aspect_classes": self.aspect_classes,
                "features_corroborate_only": self.features_corroborate_only,
                "skipped_records": self.skipped_records,
                "flag_counts": dict(self.flag_counts)}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"nodes": nodes_path, "edges": edges_path, "meta": meta_path}

    @classmethod
    def from_export(cls, out_dir: str | Path) -> "KnowledgeGraph":
        """Load a graph written by export_jsonl(). No extraction or inference
        runs, so the graph is exactly what was exported."""
        out = Path(out_dir)
        meta_path = out / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found; re-export with this version of export_jsonl()")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != EXPORT_FORMAT:
            raise ValueError(f"unsupported export format {meta.get('format')!r}")
        kg = cls()
        with open(out / "nodes.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    kg.g.add_node(d.pop("id"), **d)
        with open(out / "edges.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    u, v = d.pop("src"), d.pop("dst")
                    kg.g.add_edge(u, v, key=d["rel"], **d)
        kg.vocab_versions = dict(meta.get("vocab_versions") or {})
        kg.aspect_classes = dict(meta.get("aspect_classes") or {})
        kg.features_corroborate_only = bool(meta.get("features_corroborate_only"))
        kg.skipped_records = int(meta.get("skipped_records") or 0)
        kg.flag_counts = Counter(meta.get("flag_counts") or {})
        return kg


_PROVENANCE_KEYS = ("source", "source_fields", "confidence", "evidence",
                    "span", "rule_id", "evidence_type", "context", "evidence_polarity", "review_id")


def _evidence_item(edge: Edge) -> Dict[str, Any]:
    d = edge.to_dict()
    return {k: d.get(k) for k in _PROVENANCE_KEYS}


def _evidence_key(item: Dict[str, Any]) -> tuple:
    span = item.get("span")
    return (item.get("source"), tuple(item.get("source_fields") or ()),
            item.get("evidence"), tuple(span) if span else None, item.get("rule_id"),
            item.get("review_id"))


def iter_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream records, skipping blank or malformed lines (same policy as the
    CODE/data scripts)."""
    with open(path, "r", encoding="utf-8") as f:
        n = 0
        for line in f:
            if limit is not None and n >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
