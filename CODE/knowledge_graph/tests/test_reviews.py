"""Review evidence tests: SUPPORTS edges and review-derived quality HAS_ASPECT.

Review texts below are short test inputs (several adapted from sample
reviews). Run from CODE/:
    python -m unittest discover -s knowledge_graph/tests -t . -v
"""

import unittest
from pathlib import Path

from knowledge_graph.extractors import extract_reviews, unique_reviews
from knowledge_graph.graph import KnowledgeGraph, iter_jsonl
from knowledge_graph.ontology import (
    Edge, NodeType, Polarity, Provenance, RelType, aspect_id, product_id, review_id,
)
from knowledge_graph.review_inference import (
    AspectMention, RuleBasedReviewClassifier, make_review_aspect_extractor,
)
from knowledge_graph.vocabulary import VocabularyError, load_aspects

SAMPLE = Path(__file__).resolve().parents[3] / "DATA" / "samples" / "sample_corpus_v2.jsonl"
ASPECTS = load_aspects()
CLF = RuleBasedReviewClassifier(ASPECTS)
TS = iter(range(1600000000000, 1700000000000, 1000))


def review(text, rating=5.0, verified=True, title="", ts=None):
    return {"title": title, "text": text, "rating": rating, "helpful_vote": 0,
            "verified_purchase": verified, "timestamp": ts if ts is not None else next(TS)}


def product(reviews, domain="Toys & Games", title="Wooden Train Set", asin="B0REVTEST1",
            features=None, description=None):
    return {"parent_asin": asin, "domain": domain, "title": title,
            "categories": [domain], "features": features or [], "description": description or [],
            "details": {}, "brand_or_author": "Acme", "review_samples": reviews,
            "review_count": len(reviews)}


def mentions(text, domain="Toys & Games"):
    return {(m.label, m.polarity) for m in CLF.classify(text, domain)}


def build(records, **kw):
    return KnowledgeGraph.build(records, **kw)


def supports(kg, asin, label):
    return kg.review_evidence(asin, label)


def has_quality(kg, asin, label):
    pid, aid = product_id(asin), aspect_id(label)
    return kg.g.has_edge(pid, aid, key=RelType.HAS_ASPECT.value)


# ------------------------------------------------------------ classifier
class TestClassifierPolarity(unittest.TestCase):
    def test_positive(self):
        self.assertIn(("durability", "positive"), mentions("Very sturdy, survived my toddler."))

    def test_negative(self):
        self.assertIn(("durability", "negative"), mentions("It fell apart after two days."))

    def test_negated_positive_is_negative(self):
        self.assertEqual(mentions("It is not very sturdy."), {("durability", "negative")})

    def test_negated_negative_is_neutral_not_positive(self):
        self.assertEqual(mentions("Hasn't broken yet."), {("durability", "neutral")})
        self.assertEqual(mentions("The cards were not flimsy."), {("durability", "neutral")})

    def test_negation_exception(self):
        self.assertIn(("durability", "positive"), mentions("Not only durable but cute."))

    def test_negation_stops_at_comma(self):
        self.assertIn(("craftsmanship", "positive"), mentions("No complaints, very well made."))

    def test_hedged_and_question_are_neutral(self):
        self.assertEqual(mentions("I hope it is durable."), {("durability", "neutral")})
        self.assertEqual(mentions("Is it durable?"), {("durability", "neutral")})
        self.assertEqual(mentions("It looked like it would be very informative.", "Books"),
                         {("informative", "neutral")})

    def test_hedge_scope_stops_at_comma(self):
        self.assertEqual(mentions("If you want a science kit look elsewhere, this kit is overpriced."),
                         {("value_for_money", "negative")})

    def test_contrast_clauses_judged_separately(self):
        ms = CLF.classify("Very sturdy but the paint peeling off was sad.", "Toys & Games")
        pols = sorted(m.polarity for m in ms if m.label == "durability")
        self.assertEqual(pols, ["negative", "positive"])
        self.assertEqual(len({m.clause for m in ms}), 2)

    def test_longer_cue_wins(self):
        self.assertEqual(mentions("Not worth the money."), {("value_for_money", "negative")})

    def test_domain_scoping(self):
        self.assertEqual(mentions("Boring.", "Books"), {("engaging_read", "negative")})
        self.assertEqual(mentions("Boring.", "Toys & Games"), {("fun_engagement", "negative")})
        self.assertEqual(mentions("Very sturdy.", "Books"), set())

    def test_packaging_damage_is_not_durability(self):
        self.assertEqual(mentions("Arrived broken."), {("packaging_presentation", "negative")})
        self.assertEqual(mentions("The packaging on the toy was torn and broken."), set())

    def test_ambiguous_cues_excluded(self):
        self.assertEqual(mentions("He becomes salt and light by engaging in dialogue.", "Books"), set())
        self.assertEqual(mentions("The film lasted long enough to cover the fridge."), set())
        self.assertEqual(mentions("The Rapture is also well worth reading.", "Books"),
                         {("engaging_read", "positive")})

    def test_injuries_are_not_product_durability(self):
        # B07TVS1P8C (5-star): the child and the plush's story, not product damage.
        self.assertEqual(mentions("My daughter had broken her wrist a few days before."), set())
        self.assertEqual(mentions("She had a stuffy with a broken leg."), set())
        self.assertEqual(mentions("The leg broke off."), {("durability", "negative")})

    def test_spans_index_review_text(self):
        text = "Cute. It’s sturdy and well made!"
        for m in CLF.classify(text, "Toys & Games"):
            self.assertEqual(text[m.span[0]:m.span[1]], m.cue)
            self.assertIn(m.cue, m.clause)

    def test_empty_text(self):
        self.assertEqual(CLF.classify("", "Toys & Games"), [])


# ------------------------------------------------------- SUPPORTS edges
class TestSupportsEdges(unittest.TestCase):
    def test_edge_attributes_and_provenance(self):
        r = review("So sturdy, my kids love it. Survived many drops.", rating=4.0, verified=False)
        rec = product([r])
        kg = build([rec])
        (ev,) = supports(kg, rec["parent_asin"], "durability")
        self.assertEqual(ev["polarity"], "positive")
        self.assertEqual((ev["rating"], ev["verified_purchase"]), (4.0, False))
        rid = review_id(rec["parent_asin"], r)
        self.assertEqual(ev["review_id"], rid)
        e = kg.edge(rid, RelType.SUPPORTS, aspect_id("durability"))
        self.assertEqual(e["source"], "review_text")
        self.assertEqual(e["evidence_type"], "review")
        self.assertEqual(e["confidence"], max(x["confidence"] for x in e["evidence_list"]))
        for x in e["evidence_list"]:
            self.assertEqual(x["review_id"], rid)
            self.assertIn(x["evidence_polarity"], ("positive", "negative", "neutral"))
            s, t = x["span"]
            self.assertIn(r["text"][s:t], x["evidence"])
        self.assertEqual(kg.validate(), [])

    def test_supports_reuse_review_node_ids(self):
        rec = product([review("Very durable."), review("Broke in a day.", rating=1.0)])
        built = {n.id for n in extract_reviews(rec).nodes}
        kg = build([rec])
        srcs = {u for u, _, k in kg.g.edges(keys=True) if k == "SUPPORTS"}
        self.assertEqual(srcs, built)

    def test_mixed_review(self):
        rec = product([review("Sturdy at first but it broke after a week.", rating=3.0)])
        kg = build([rec])
        (ev,) = supports(kg, rec["parent_asin"], "durability")
        self.assertEqual(ev["polarity"], "mixed")
        self.assertEqual(sorted(x["polarity"] for x in ev["evidence"]), ["negative", "positive"])

    def test_negative_review(self):
        rec = product([review("Flimsy plastic, snapped the first day.", rating=1.0)])
        kg = build([rec])
        (ev,) = supports(kg, rec["parent_asin"], "durability")
        self.assertEqual(ev["polarity"], "negative")

    def test_star_rating_alone_is_not_evidence(self):
        rec = product([review("Arrived Tuesday.", rating=5.0), review("Meh.", rating=1.0),
                       review("", rating=5.0)])
        kg = build([rec])
        self.assertEqual(kg.review_evidence(rec["parent_asin"]), [])
        self.assertEqual(kg.aspects_of(rec["parent_asin"], aspect_class="quality"), [])

    def test_seller_text_is_never_review_evidence(self):
        rec = product([], title="Durable, sturdy, well made train - great gift",
                      features=["Very durable and high quality", "Kids love it, a big hit"],
                      description=["Built to last. Easy to use."])
        kg = build([rec])
        self.assertEqual(kg.review_evidence(rec["parent_asin"]), [])
        self.assertEqual(kg.aspects_of(rec["parent_asin"], aspect_class="quality"), [])
        self.assertFalse(any(k == "SUPPORTS" for _, _, k in kg.g.edges(keys=True)))

    def test_missing_review_field(self):
        rec = product([])
        del rec["review_samples"]
        kg = build([rec])
        self.assertEqual(kg.reviews_of(rec["parent_asin"]), [])
        self.assertEqual(kg.review_concerns_of(rec["parent_asin"]), {})
        self.assertEqual(kg.validate(), [])

    def test_malformed_reviews_tolerated(self):
        rec = product([None, "text", {"rating": 5.0}, {"text": "   "},
                       review("Very durable.")])
        kg = build([rec])
        self.assertEqual(len(kg.reviews_of(rec["parent_asin"])), 1)
        self.assertEqual(kg.validate(), [])

    def test_classifier_is_replaceable(self):
        class Stub:
            version = "stub@0"

            def classify(self, text, domain):
                return [AspectMention("durability", "positive", 0.9, text[:4], (0, 4), text, text,
                                      "direct", "positive")]
        rec = product([review("abcd one"), review("abcd two")])
        kg = build([rec], review_classifier=Stub())
        self.assertTrue(has_quality(kg, rec["parent_asin"], "durability"))
        e = kg.edge(product_id(rec["parent_asin"]), RelType.HAS_ASPECT, aspect_id("durability"))
        self.assertEqual(e["classifier"], "stub@0")
        self.assertEqual(kg.validate(), [])


# ------------------------------------------------------- aggregation
class TestReviewAggregation(unittest.TestCase):
    def test_two_independent_positives_create_has_aspect(self):
        rec = product([review("Very sturdy."), review("So durable, survived my toddler.")])
        kg = build([rec])
        self.assertTrue(has_quality(kg, rec["parent_asin"], "durability"))
        (ra,) = kg.review_aspects_of(rec["parent_asin"])
        self.assertEqual((ra["label"], ra["n_positive"], ra["n_negative"]), ("durability", 2, 0))
        e = kg.edge(product_id(rec["parent_asin"]), RelType.HAS_ASPECT, aspect_id("durability"))
        self.assertEqual((e["source"], e["evidence_type"], e["polarity"]),
                         ("review_aggregate", "review", "positive"))
        self.assertEqual(len({x["review_id"] for x in e["evidence_list"]}), 2)
        self.assertEqual(kg.validate(), [])

    def test_single_positive_is_insufficient(self):
        rec = product([review("Very sturdy."), review("Cute colors.")])
        kg = build([rec])
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))
        signal = kg.node(product_id(rec["parent_asin"]))["review_aspect_signals"]["durability"]
        self.assertEqual(signal["status"], "insufficient")

    def test_duplicate_reviews_count_once(self):
        r = review("Very sturdy.", ts=1600000000000)
        rec = product([r, dict(r)])                    # exact duplicate: same Review ID
        kg = build([rec])
        self.assertEqual(len(kg.reviews_of(rec["parent_asin"])), 1)
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))

    def test_identical_text_counts_once(self):
        rec = product([review("Very sturdy.", ts=1), review("Very sturdy.", ts=2)])
        kg = build([rec])
        self.assertEqual(len(kg.reviews_of(rec["parent_asin"])), 2)  # distinct Review nodes
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))

    def test_negative_evidence_never_becomes_has_aspect(self):
        rec = product([review("Broke in a day.", 1.0), review("Flimsy, fell apart.", 1.0),
                       review("Cheaply made and broke.", 2.0)])
        kg = build([rec])
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))
        concerns = kg.review_concerns_of(rec["parent_asin"])
        self.assertEqual(concerns["durability"]["negative"], 3)
        self.assertEqual(kg.validate(), [])

    def test_contradictory_reviews_block_support(self):
        rec = product([review("Very sturdy."), review("So durable."),
                       review("It broke right away.", 2.0), review("Snapped in half.", 1.0)])
        kg = build([rec])
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))
        signal = kg.node(product_id(rec["parent_asin"]))["review_aspect_signals"]["durability"]
        self.assertEqual((signal["positive"], signal["negative"], signal["status"]),
                         (2, 2, "insufficient"))

    def test_mixed_reviews_count_against(self):
        rec = product([review("Very sturdy."), review("So durable."),
                       review("Sturdy but the wheel broke off.", 3.0)])
        kg = build([rec])
        signal = kg.node(product_id(rec["parent_asin"]))["review_aspect_signals"]["durability"]
        self.assertEqual(signal["mixed"], 1)
        self.assertTrue(has_quality(kg, rec["parent_asin"], "durability"))  # 2 >= 2 * 1
        e = kg.edge(product_id(rec["parent_asin"]), RelType.HAS_ASPECT, aspect_id("durability"))
        self.assertEqual(e["n_mixed"], 1)
        self.assertEqual(len(e["evidence_list"]), 2)  # the mixed review is not cited

    def test_rating_conflict_excluded(self):
        rec = product([review("Very sturdy.", 1.0), review("Durable I guess.", 2.0),
                       review("So durable.", 5.0)])
        kg = build([rec])
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))
        conflicts = [e for e in kg.review_evidence(rec["parent_asin"], "durability") if e["rating_conflict"]]
        self.assertEqual(len(conflicts), 2)

    def test_neutral_reviews_do_not_support(self):
        rec = product([review("Hope it is durable."), review("Is it sturdy?"), review("Not flimsy.")])
        kg = build([rec])
        self.assertFalse(has_quality(kg, rec["parent_asin"], "durability"))
        signal = kg.node(product_id(rec["parent_asin"]))["review_aspect_signals"]["durability"]
        self.assertEqual(signal["status"], "neutral_only")


# ------------------------------------------------------- validation
class TestReviewValidation(unittest.TestCase):
    def setUp(self):
        self.rec = product([review("Very sturdy."), review("So durable."),
                            review("It broke.", 1.0)])
        self.kg = build([self.rec])
        self.pid = product_id(self.rec["parent_asin"])
        self.aid = aspect_id("durability")

    def test_valid(self):
        self.assertEqual(self.kg.validate(), [])

    def test_negative_review_cannot_be_cited_as_support(self):
        neg = [e for e in self.kg.review_evidence(self.rec["parent_asin"], "durability")
               if e["polarity"] == "negative"][0]
        self.kg.add_edge(Edge(self.pid, RelType.HAS_ASPECT, self.aid, Provenance(
            "review_aggregate", ["review_samples.text"], 0.7, "It broke.", (0, 8),
            rule_id="durability.review_aggregate", evidence_type="review",
            evidence_polarity="positive", review_id=neg["review_id"])))
        self.assertTrue(any("not clean positive support" in p for p in self.kg.validate()))

    def test_listing_evidence_cannot_support_quality(self):
        rec = product([], asin="B0REVTEST2")
        kg = build([rec])
        kg.add_edge(Edge(product_id("B0REVTEST2"), RelType.HAS_ASPECT, self.aid, Provenance(
            "feature_lexicon", ["features[0]"], 0.5, "durable", rule_id="durability.features",
            evidence_type="listing_stated"), {"vocab_version": ASPECTS.version_tag}))
        self.assertTrue(any("quality aspect without review evidence" in p for p in kg.validate()))

    def test_supports_polarity_must_match_items(self):
        rid = next(e["review_id"] for e in self.kg.review_evidence(self.rec["parent_asin"], "durability")
                   if e["polarity"] == "negative")
        self.kg.g.edges[rid, self.aid, "SUPPORTS"]["polarity"] = "positive"
        self.assertTrue(any("polarity does not match" in p for p in self.kg.validate()))

    def test_item_polarity_mixed_rejected(self):
        with self.assertRaises(ValueError):
            Provenance("review_text", ["review_samples.text"], 0.7, evidence_polarity="mixed")


class TestReviewVocabulary(unittest.TestCase):
    def test_all_quality_aspects_have_cues(self):
        self.assertEqual(len(ASPECTS.reserved), 12)
        for c in ASPECTS.reserved:
            self.assertIsNotNone(c.review_cues, c.label)
            self.assertEqual(c.rules, [])  # still no listing rules

    def test_min_positive_reviews_at_least_two(self):
        import json
        import tempfile
        from knowledge_graph.vocabulary import ASPECTS_PATH
        data = json.loads(Path(ASPECTS_PATH).read_text(encoding="utf-8"))
        data["review_inference"]["aggregation"]["min_positive_reviews"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "aspects.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(VocabularyError):
                load_aspects(p)


class TestVocabularyDocs(unittest.TestCase):
    def test_readme_is_generated_from_current_json(self):
        from knowledge_graph.vocab_docs import README_PATH, render
        self.assertEqual(README_PATH.read_text(encoding="utf-8"), render(),
                         "vocab/README.md is stale: run python -m knowledge_graph.vocab_docs")


class TestReviewReports(unittest.TestCase):
    def setUp(self):
        self.rec = product([review("Very sturdy."), review("So durable."), review("It broke.", 1.0)])
        self.kg = build([self.rec])

    def test_review_coverage(self):
        from knowledge_graph.reports import coverage
        cov = coverage(self.kg)["Toys & Games"]
        self.assertIn("match_type", cov["interests"])
        self.assertEqual(cov["aspects"]["by_label"].get("durability"), None)  # attribute-only
        r = cov["reviews"]
        self.assertEqual((r["products_with_reviews"], r["reviews"]), (1, 3))
        self.assertEqual(r["supports_edges_by_polarity"], {"positive": 2, "negative": 1})
        self.assertEqual(r["review_aspects_by_label"], {"durability": 1})

    def test_review_sheet_and_predictions_never_overwrite(self):
        import csv
        import tempfile
        from knowledge_graph.reports import (
            write_labeling_sheet, write_review_sheet, write_sheet_predictions,
        )
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / "review.csv"
            self.assertEqual(write_review_sheet(self.kg, sheet), 3)
            with self.assertRaises(FileExistsError):
                write_review_sheet(self.kg, sheet)
            with open(sheet, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["predicted_item_polarity"] for r in rows}, {"positive", "negative"})
            labels = Path(tmp) / "labels.csv"
            write_labeling_sheet(self.kg, labels, per_domain=1)
            before = labels.read_bytes()
            with self.assertRaises(ValueError):
                write_sheet_predictions(self.kg, labels, labels)
            out = Path(tmp) / "pred.csv"
            write_sheet_predictions(self.kg, labels, out)
            self.assertEqual(labels.read_bytes(), before)


@unittest.skipUnless(SAMPLE.exists(), "development sample not found")
class TestSampleReviews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = KnowledgeGraph.build(iter_jsonl(SAMPLE))

    def test_valid(self):
        self.assertEqual(self.kg.validate(), [])

    def test_review_nodes_match_unique_reviews(self):
        expected = 0
        for rec in iter_jsonl(SAMPLE):
            expected += len(unique_reviews(rec)[0])
        self.assertEqual(len(self.kg.nodes_of_type(NodeType.REVIEW)), expected)
        about = sum(1 for _, _, k in self.kg.g.edges(keys=True) if k == "ABOUT")
        self.assertEqual(about, expected)

    def test_every_supports_has_evidence_text(self):
        n = 0
        for u, _, k, d in self.kg.g.edges(keys=True, data=True):
            if k == "SUPPORTS":
                n += 1
                text = self.kg.g.nodes[u]["text"]
                for x in d["evidence_list"]:
                    self.assertIn(text[x["span"][0]:x["span"][1]], x["evidence"])
        self.assertGreater(n, 0)

    def test_review_has_aspect_is_positive_only(self):
        n = 0
        for _, v, k, d in self.kg.g.edges(keys=True, data=True):
            if k == "HAS_ASPECT" and self.kg.g.nodes[v]["aspect_class"] == "quality":
                n += 1
                self.assertEqual(d["polarity"], Polarity.POSITIVE.value)
                self.assertGreaterEqual(d["n_positive"], 2)
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
