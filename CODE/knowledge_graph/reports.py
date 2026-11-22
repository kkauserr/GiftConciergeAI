"""Coverage statistics, suspicious-match review lists, and labeling sheets."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .graph import KnowledgeGraph
from .ontology import (
    ASPECT_ATTRIBUTE, ASPECT_QUALITY, SOURCE_FEATURE_LEXICON, SOURCE_TITLE_LEXICON,
    NodeType, RelType,
)

DOMAINS = ("Books", "Handmade Products", "Toys & Games")


def _products_by_domain(kg: KnowledgeGraph) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for p in kg.nodes_of_type(NodeType.PRODUCT):
        out[kg.g.nodes[p].get("domain")].append(p)
    return out


def coverage(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Per-domain coverage of MATCHES, attribute HAS_ASPECT and review evidence.

    For "interests" and "aspects" (attribute aspects only, comparable with
    earlier versions): products with at least one edge, mean edges per
    covered product, per-label product counts, and the source of each
    edge's highest-confidence evidence. "reviews" summarises Review nodes,
    SUPPORTS polarity, review-derived quality HAS_ASPECT and concerns.
    """
    report: Dict[str, Any] = {}
    for domain, prods in sorted(_products_by_domain(kg).items(), key=lambda x: str(x[0])):
        entry: Dict[str, Any] = {"products": len(prods)}
        for rel, key, cls in ((RelType.MATCHES, "interests", None),
                              (RelType.HAS_ASPECT, "aspects", ASPECT_ATTRIBUTE)):
            covered, n_edges = 0, 0
            labels: Counter = Counter()
            best_source: Counter = Counter()
            for p in prods:
                outs = [(v, d) for _, v, k, d in kg.g.out_edges(p, keys=True, data=True)
                        if k == rel.value and (cls is None or kg.aspect_classes.get(v) == cls)]
                if outs:
                    covered += 1
                    n_edges += len(outs)
                for v, d in outs:
                    labels[kg.g.nodes[v]["label"]] += 1
                    best_source[d["source"]] += 1
            entry[key] = {
                "products_covered": covered,
                "coverage_pct": round(100 * covered / len(prods), 1) if prods else 0.0,
                "mean_per_covered_product": round(n_edges / covered, 2) if covered else 0.0,
                "edges": n_edges,
                "best_evidence_source": dict(best_source.most_common()),
                "by_label": dict(labels.most_common()),
            }
            if rel == RelType.MATCHES:
                entry[key]["match_type"] = dict(Counter(
                    d.get("match_type") for p in prods
                    for _, _, k, d in kg.g.out_edges(p, keys=True, data=True) if k == rel.value))
        entry["reviews"] = review_coverage(kg, prods)
        report[domain] = entry
    return report


def review_coverage(kg: KnowledgeGraph, prods: List[str]) -> Dict[str, Any]:
    """Review evidence for a set of product node IDs."""
    with_reviews, n_reviews, verified = 0, 0, 0
    supports: Counter = Counter()
    items: Counter = Counter()
    supported_products = 0
    supported: Counter = Counter()
    concerns: Counter = Counter()
    for p in prods:
        rids = kg._in(p, RelType.ABOUT)
        n_reviews += len(rids)
        with_reviews += bool(rids)
        for rid in rids:
            verified += bool(kg.g.nodes[rid].get("verified_purchase"))
            for _, _, k, d in kg.g.out_edges(rid, keys=True, data=True):
                if k == RelType.SUPPORTS.value:
                    supports[d["polarity"]] += 1
                    for x in d["evidence_list"]:
                        items[x["evidence_polarity"]] += 1
        q = [kg.g.nodes[v]["label"] for _, v, k in kg.g.out_edges(p, keys=True)
             if k == RelType.HAS_ASPECT.value and kg.aspect_classes.get(v) == ASPECT_QUALITY]
        supported_products += bool(q)
        supported.update(q)
        for label, sig in (kg.g.nodes[p].get("review_aspect_signals") or {}).items():
            if sig.get("status") == "concern":
                concerns[label] += 1
    return {
        "products_with_reviews": with_reviews,
        "reviews": n_reviews,
        "verified_reviews": verified,
        "supports_edges_by_polarity": dict(supports.most_common()),
        "evidence_items_by_polarity": dict(items.most_common()),
        "products_with_review_aspects": supported_products,
        "review_aspect_coverage_pct": round(100 * supported_products / len(prods), 1) if prods else 0.0,
        "review_has_aspect_edges": sum(supported.values()),
        "review_aspects_by_label": dict(supported.most_common()),
        "review_concerns_by_label": dict(concerns.most_common()),
    }


def suspicious(kg: KnowledgeGraph) -> List[Dict[str, Any]]:
    """Inferred edges worth a manual look, each with a reason.

    Heuristics:
      text_only_low     - only feature-text evidence (weakest tier)
      books_title_only  - Books interest from a title phrase, no category support
      many_interests    - product has 4 or more interests
      many_materials    - product has 3 or more material aspects
      review_supported_despite_negatives - review-derived quality aspect
                          that also has negative or mixed reviews
      review_rating_conflict - positive review text in a 1-2 star review
      review_mixed      - one review with both positive and negative items
    """
    out = []
    for p in kg.nodes_of_type(NodeType.PRODUCT):
        node = kg.g.nodes[p]
        asin, domain = node["parent_asin"], node.get("domain")
        interests = kg.interests_of(asin)
        aspects = kg.aspects_of(asin, aspect_class=ASPECT_ATTRIBUTE)
        for ra in kg.review_aspects_of(asin):
            if ra["n_negative"] or ra["n_mixed"]:
                out.append({"parent_asin": asin, "domain": domain, "title": node.get("title"),
                            "reason": "review_supported_despite_negatives", "label": ra["label"],
                            "counts": {k: ra[k] for k in ("n_positive", "n_negative", "n_mixed")}})
        for ev in kg.review_evidence(asin):
            if ev["rating_conflict"] or ev["polarity"] == "mixed":
                out.append({"parent_asin": asin, "domain": domain, "title": node.get("title"),
                            "reason": "review_rating_conflict" if ev["rating_conflict"] else "review_mixed",
                            "label": ev["aspect"], "rating": ev["rating"],
                            "evidence": [x["text"] for x in ev["evidence"]]})
        for rel, items in (("MATCHES", interests), ("HAS_ASPECT", aspects)):
            for it in items:
                if it["sources"] == [SOURCE_FEATURE_LEXICON]:
                    out.append(_flag(asin, domain, node, rel, it, "text_only_low"))
                elif rel == "MATCHES" and domain == "Books" and it["sources"] == [SOURCE_TITLE_LEXICON]:
                    out.append(_flag(asin, domain, node, rel, it, "books_title_only"))
        if len(interests) >= 4:
            out.append({"parent_asin": asin, "domain": domain, "title": node.get("title"),
                        "reason": "many_interests",
                        "labels": [i["label"] for i in interests]})
        mats = [a["label"] for a in aspects if a["label"].startswith("material_")]
        if len(mats) >= 3:
            out.append({"parent_asin": asin, "domain": domain, "title": node.get("title"),
                        "reason": "many_materials", "labels": mats})
    return out


def _flag(asin, domain, node, rel, item, reason):
    ev = item["evidence"][0]
    return {"parent_asin": asin, "domain": domain, "title": node.get("title"),
            "reason": reason, "relation": rel, "label": item["label"],
            "confidence": item["confidence"], "evidence": ev.get("evidence"),
            "context": ev.get("context")}


# --------------------------------------------------------------------------
# Labeling sheet
# --------------------------------------------------------------------------

SHEET_COLUMNS = [
    "sheet_id", "parent_asin", "domain", "selection_stratum", "title", "category_path",
    "brand", "predicted_interests", "predicted_aspects",
    # annotator columns
    "gold_interests", "wrong_predicted_interests", "missed_interests",
    "wrong_predicted_aspects", "missed_aspects", "annotator", "notes",
]


def _fmt(items: List[Dict[str, Any]]) -> str:
    parts = []
    for it in items:
        ev = it["evidence"][0]
        src = "+".join(it["sources"])
        parts.append(f"{it['label']} ({it['confidence']:.2f}; {src}; \"{ev.get('evidence')}\")")
    return " | ".join(parts)


def select_for_labeling(kg: KnowledgeGraph, per_domain: int = 20, seed: str = "kg-label-v1"):
    """Deterministic, balanced sample: per domain, half the products with at
    least one predicted interest (to measure precision) and half with none
    (to look for misses). Order is by sha1(seed + asin), so it is
    reproducible and independent of file order."""
    chosen = []
    by_domain = _products_by_domain(kg)
    for domain in DOMAINS:
        prods = sorted(by_domain.get(domain, []),
                       key=lambda p: hashlib.sha1(f"{seed}:{p}".encode()).hexdigest())
        with_i = [p for p in prods if kg.interests_of(kg.g.nodes[p]["parent_asin"])]
        without = [p for p in prods if p not in set(with_i)]
        half = per_domain // 2
        take_with = with_i[:max(half, per_domain - len(without))]
        take_without = without[:per_domain - len(take_with)]
        chosen += [(domain, "has_prediction", p) for p in take_with]
        chosen += [(domain, "no_prediction", p) for p in take_without]
    return chosen


REVIEW_SHEET_COLUMNS = [
    "sheet_id", "parent_asin", "domain", "title", "review_id", "rating", "verified_purchase",
    "aspect", "predicted_item_polarity", "predicted_review_polarity", "rule_id", "evidence_clause",
    "full_review_text", "in_review_aspect_has_aspect",
    # annotator columns
    "gold_aspect_correct", "gold_polarity", "notes", "annotator",
]


def select_review_evidence(kg: KnowledgeGraph, per_cell: int = 4,
                           seed: str = "kg-review-label-v1") -> List[Dict[str, Any]]:
    """Deterministic sample of SUPPORTS evidence items, stratified by
    (aspect, item polarity): up to `per_cell` items per cell, so rare
    negative and neutral evidence is represented, not only the dominant
    positive cues."""
    cells: Dict[tuple, List[tuple]] = defaultdict(list)
    for rid, aid, k, d in kg.g.edges(keys=True, data=True):
        if k != RelType.SUPPORTS.value:
            continue
        for x in d["evidence_list"]:
            key = hashlib.sha1(f"{seed}|{rid}|{aid}|{x['span']}".encode()).hexdigest()
            cells[(kg.g.nodes[aid]["label"], x["evidence_polarity"])].append((key, rid, aid, d, x))
    chosen = []
    for cell in sorted(cells):
        chosen += sorted(cells[cell])[:per_cell]
    return chosen


def write_review_sheet(kg: KnowledgeGraph, path: str | Path, per_cell: int = 4,
                       overwrite: bool = False) -> int:
    """Write the review-evidence labeling sheet (refuses to overwrite)."""
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; it may contain annotations. "
                              "Choose a new path or pass overwrite=True (--force).")
    rows = select_review_evidence(kg, per_cell)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_SHEET_COLUMNS)
        w.writeheader()
        for i, (_, rid, aid, d, x) in enumerate(rows, start=1):
            pid = kg._out(rid, RelType.ABOUT)[0]
            prod, rev = kg.g.nodes[pid], kg.g.nodes[rid]
            w.writerow({
                "sheet_id": f"R{i:03d}", "parent_asin": prod["parent_asin"], "domain": prod["domain"],
                "title": prod["title"], "review_id": rid, "rating": rev.get("rating"),
                "verified_purchase": rev.get("verified_purchase"),
                "aspect": kg.g.nodes[aid]["label"], "predicted_item_polarity": x["evidence_polarity"],
                "predicted_review_polarity": d["polarity"], "rule_id": x["rule_id"],
                "evidence_clause": x["evidence"], "full_review_text": rev.get("text"),
                "in_review_aspect_has_aspect": kg.g.has_edge(pid, aid, key=RelType.HAS_ASPECT.value),
            })
    return len(rows)


PREDICTION_COLUMNS = ["sheet_id", "parent_asin", "domain", "vocab_versions",
                      "predicted_interests", "predicted_aspects", "review_aspects", "review_concerns"]


def write_sheet_predictions(kg: KnowledgeGraph, sheet: str | Path, path: str | Path,
                            overwrite: bool = False) -> int:
    """Current predictions for the products of an existing labeling sheet,
    written to a separate file. The sheet itself (and any annotations in
    it) is only read, so gold labels collected on an older sheet can be
    scored against the current vocabulary version."""
    if Path(path).resolve() == Path(sheet).resolve():
        raise ValueError("predictions must go to a different file than the sheet")
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Choose a new path or pass overwrite=True (--force).")
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    versions = ";".join(f"{v}" for _, v in sorted(kg.vocab_versions.items()))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        w.writeheader()
        for r in rows:
            asin = r["parent_asin"]
            s = kg.product_summary(asin)
            w.writerow({
                "sheet_id": r["sheet_id"], "parent_asin": asin, "domain": r["domain"],
                "vocab_versions": versions,
                "predicted_interests": " | ".join(
                    f"{i['label']} ({i['confidence']:.2f}; {i['match_type']}; {'+'.join(i['sources'])})"
                    for i in s["interests"]),
                "predicted_aspects": _fmt(s["aspects"]),
                "review_aspects": " | ".join(
                    f"{a['label']} (+{a['n_positive']}/-{a['n_negative']}/~{a['n_mixed']} of {a['n_reviews']})"
                    for a in s["review_aspects"]),
                "review_concerns": " | ".join(
                    f"{k} (-{v['negative']}/+{v['positive']})" for k, v in s["review_concerns"].items()),
            })
    return len(rows)


def write_labeling_sheet(kg: KnowledgeGraph, path: str | Path, per_domain: int = 20,
                         overwrite: bool = False) -> int:
    """Write the balanced labeling sheet. Refuses to replace an existing
    file (it may hold annotations) unless overwrite=True."""
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; it may contain annotations. "
                              "Choose a new path or pass overwrite=True (--force).")
    rows = select_for_labeling(kg, per_domain)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens accented titles correctly
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for i, (domain, stratum, pid) in enumerate(rows, start=1):
            asin = kg.g.nodes[pid]["parent_asin"]
            s = kg.product_summary(asin)
            w.writerow({
                "sheet_id": f"L{i:03d}", "parent_asin": asin, "domain": domain,
                "selection_stratum": stratum, "title": s["title"],
                "category_path": " > ".join(s["category_path"]),
                "brand": s["brand"]["name"] if s["brand"] else "",
                "predicted_interests": _fmt(s["interests"]),
                "predicted_aspects": _fmt(s["aspects"]),
            })
    return len(rows)
