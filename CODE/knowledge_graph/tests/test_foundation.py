"""Foundation tests: ontology IDs, normalisation, direct extractors, graph.

Unit tests use small inline fixture records modelled on real sample cases.
These are test inputs only and never written to the KG outputs. Integration
tests run on DATA/samples/sample_corpus_v2.jsonl and skip if it is missing.

Run from CODE/:  python -m unittest discover -s knowledge_graph/tests -t . -v
"""

import unittest
from pathlib import Path

from knowledge_graph import normalize as nz
from knowledge_graph.extractors import (
    RecordError, extract_brand, extract_categories, extract_product,
    extract_record, extract_reviews,
)
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.ontology import (
    NodeType, Polarity, Provenance, RelType, brand_id, category_id,
    product_id, review_id, slugify,
)

SAMPLE = Path(__file__).resolve().parents[3] / "DATA" / "samples" / "sample_corpus_v2.jsonl"


def rec(**kw):
    base = {"parent_asin": "B000TEST01", "domain": "Toys & Games", "title": "Test toy",
            "brand_or_author": "LEGO", "store": "LEGO",
            "categories": ["Toys & Games", "Building Toys", "Building Sets"],
            "main_category": "Toys & Games", "specific_category": "Building Sets",
            "details": {}, "age_min": None, "age_max": None}
    base.update(kw)
    return base


class TestOntology(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("LEGO"), slugify("Lego"))
        self.assertEqual(slugify("Rubie's"), "rubies")
        self.assertEqual(slugify("Toys & Games"), "toys-and-games")
        self.assertEqual(slugify("Café"), "cafe")

    def test_category_id_is_path_based(self):
        a = category_id(["Books", "Literature & Fiction"])
        b = category_id(["Books", "Children's Books", "Literature & Fiction"])
        self.assertNotEqual(a, b)
        with self.assertRaises(ValueError):
            category_id([])

    def test_provenance_confidence_bounds(self):
        with self.assertRaises(ValueError):
            Provenance("metadata", ["x"], confidence=1.5)

    def test_polarity_values(self):
        # "mixed" is edge-level only: a review whose items for one aspect disagree.
        self.assertEqual({p.value for p in Polarity}, {"positive", "negative", "neutral", "mixed"})


class TestReviewIds(unittest.TestCase):
    R = {"title": "Great", "text": "Kids love it.", "rating": 5.0,
         "helpful_vote": 0, "verified_purchase": True, "timestamp": 1600000000000}

    def test_deterministic(self):
        self.assertEqual(review_id("A1", self.R), review_id("A1", dict(self.R)))

    def test_scoped_to_product(self):
        self.assertNotEqual(review_id("A1", self.R), review_id("A2", self.R))

    def test_whitespace_and_case_insensitive_text(self):
        other = dict(self.R, text="  kids   LOVE it. ")
        self.assertEqual(review_id("A1", self.R), review_id("A1", other))

    def test_mutable_fields_do_not_change_identity(self):
        other = dict(self.R, helpful_vote=12, rating=4.0)
        self.assertEqual(review_id("A1", self.R), review_id("A1", other))

    def test_distinct_timestamp_distinct_id(self):
        other = dict(self.R, timestamp=1600000000001)
        self.assertNotEqual(review_id("A1", self.R), review_id("A1", other))

    def test_uses_user_id_when_present(self):
        a = dict(self.R, user_id="U1")
        b = dict(self.R, user_id="U2")
        self.assertNotEqual(review_id("A1", a), review_id("A1", b))

    def test_missing_vs_empty_field_equivalent(self):
        self.assertEqual(review_id("A1", dict(self.R, user_id="")), review_id("A1", self.R))


class TestNormalize(unittest.TestCase):
    def test_placeholder_brand_no_fallback(self):
        r = rec(brand_or_author="Generic", details={"Manufacturer": "Chainsaw Man"})
        self.assertEqual(nz.resolve_brand(r), (None, "brand_or_author", "placeholder"))

    def test_visit_store_artifact(self):
        self.assertEqual(nz.normalize_brand_name("Visit the Funko Store"), "Funko")

    def test_role_suffix(self):
        self.assertEqual(nz.normalize_brand_name("Sandra Wilde (Editor)"), "Sandra Wilde")

    def test_fallback_to_details_brand(self):
        r = rec(brand_or_author=None, details={"Brand": "Mattel"})
        self.assertEqual(nz.resolve_brand(r), ("Mattel", "details.Brand", "ok"))

    def test_missing_brand(self):
        r = rec(brand_or_author=None, details={"Department": "womens"})
        self.assertEqual(nz.resolve_brand(r), (None, None, "missing"))

    def test_brand_kind(self):
        self.assertEqual(nz.brand_kind("Books", "brand_or_author"), "author")
        self.assertEqual(nz.brand_kind("Toys & Games", "brand_or_author"), "brand")

    def test_category_path_and_fallback(self):
        self.assertEqual(nz.normalize_category_path(rec())[1], "categories")
        r = rec(domain="Handmade Products", categories=[], main_category="Handmade")
        self.assertEqual(nz.normalize_category_path(r), (["Handmade Products"], "domain"))
        self.assertEqual(nz.normalize_category_path(rec(domain="X", categories=[])), ([], "none"))

    def test_category_path_not_rerooted(self):
        path = ["Remote & App Controlled Vehicles & Parts", "Parts"]
        self.assertEqual(nz.normalize_category_path(rec(categories=path))[0], path)

    def test_age_range_valid(self):
        self.assertFalse(nz.age_range_valid(36.0, 7.0))
        self.assertTrue(nz.age_range_valid(3.0, 7.0))
        self.assertIsNone(nz.age_range_valid(3.0, None))

    def test_encoding_issue(self):
        self.assertTrue(nz.has_encoding_issue("D�cor"))
        self.assertFalse(nz.has_encoding_issue("Decor"))


class TestExtractors(unittest.TestCase):
    def test_product_requires_asin(self):
        with self.assertRaises(RecordError):
            extract_product(rec(parent_asin=""))

    def test_product_has_no_review_fields_flag(self):
        node = extract_product(rec()).nodes[0]
        self.assertEqual(node.type, NodeType.PRODUCT)
        self.assertFalse(node.attrs["has_review_fields"])
        self.assertNotIn("review_count", node.attrs)

    def test_brand_edge_provenance(self):
        ex = extract_brand(rec())
        (edge,) = ex.edges
        self.assertEqual((edge.rel, edge.dst), (RelType.MADE_BY, brand_id("LEGO")))
        self.assertEqual(edge.provenance.source, "metadata")
        self.assertEqual(edge.provenance.source_fields, ("brand_or_author",))
        self.assertEqual(edge.provenance.evidence, "LEGO")

    def test_brand_placeholder_no_edge(self):
        ex = extract_brand(rec(brand_or_author="Generic"))
        self.assertEqual((ex.nodes, ex.edges), ([], []))
        self.assertEqual(ex.flags["brand_status"], "placeholder")

    def test_categories(self):
        ex = extract_categories(rec())
        self.assertEqual(len(ex.nodes), 3)
        rels = [e.rel for e in ex.edges]
        self.assertEqual(rels.count(RelType.SUBCATEGORY_OF), 2)
        belongs = [e for e in ex.edges if e.rel == RelType.BELONGS_TO]
        self.assertEqual(belongs[0].dst, category_id(["Toys & Games", "Building Toys", "Building Sets"]))

    def test_reviews_absent_fabricates_nothing(self):
        ex = extract_reviews(rec())
        self.assertEqual((ex.nodes, ex.edges), ([], []))
        self.assertFalse(ex.flags["reviews_available"])

    def test_reviews_present_with_duplicates_and_empty(self):
        r1 = {"title": "Fun", "text": "Sturdy pieces.", "rating": 5, "timestamp": 1}
        dup = dict(r1, text="sturdy  pieces.", helpful_vote=3)
        empty = {"title": "x", "text": "  ", "rating": 1, "timestamp": 2}
        r2 = {"title": "Meh", "text": "Broke fast.", "rating": 2, "timestamp": 3}
        ex = extract_reviews(rec(review_samples=[r1, dup, empty, r2]))
        self.assertEqual(len(ex.nodes), 2)
        self.assertEqual(ex.nodes[0].attrs["duplicate_count"], 2)
        self.assertEqual(ex.flags["reviews_skipped"], 1)
        self.assertTrue(all(e.rel == RelType.ABOUT and e.dst == product_id("B000TEST01")
                            for e in ex.edges))


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraph.from_records([
            rec(),
            rec(parent_asin="B000TEST02", brand_or_author="Lego",
                categories=["Toys & Games", "Building Toys", "Building Sets"]),
            rec(parent_asin="B000TEST03", brand_or_author="Generic",
                categories=["Toys & Games", "Puzzles"]),
            rec(parent_asin=""),  # skipped
        ])

    def test_merging_and_counts(self):
        s = self.kg.stats()
        self.assertEqual(s["nodes"], {"Brand": 1, "Category": 4, "Product": 3})
        self.assertEqual(s["edges"]["MADE_BY"], 2)
        self.assertEqual(s["edges"]["SUBCATEGORY_OF"], 3)
        self.assertEqual(s["skipped_records"], 1)
        self.assertEqual(self.kg.node(brand_id("LEGO"))["aliases"], ["Lego"])

    def test_queries(self):
        self.assertEqual(self.kg.category_path_of("B000TEST01"),
                         ["Toys & Games", "Building Toys", "Building Sets"])
        root = category_id(["Toys & Games"])
        self.assertEqual(len(self.kg.products_in_category(root)), 3)
        self.assertEqual(len(self.kg.products_in_category(root, recursive=False)), 0)
        self.assertEqual(self.kg.products_by_brand(brand_id("LEGO")),
                         [product_id("B000TEST01"), product_id("B000TEST02")])
        self.assertIsNone(self.kg.brand_of("B000TEST03"))

    def test_validate(self):
        self.assertEqual(self.kg.validate(), [])

    def test_duplicate_product_record_flagged(self):
        kg = KnowledgeGraph.from_records([rec(), rec(title="dup")])
        self.assertEqual(kg.stats()["flags"]["duplicate_product_records"], 1)
        self.assertEqual(kg.node(product_id("B000TEST01"))["title"], "Test toy")

    def test_reviews_wired_into_graph(self):
        kg = KnowledgeGraph.from_records([rec(review_samples=[
            {"title": "t", "text": "Nice", "rating": 5, "timestamp": 1}])])
        self.assertEqual(len(kg.reviews_of("B000TEST01")), 1)
        self.assertEqual(kg.validate(), [])


@unittest.skipUnless(SAMPLE.exists(), "development sample not found")
class TestSampleCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = KnowledgeGraph.from_jsonl(SAMPLE)

    def test_every_product_present_and_categorised(self):
        prods = self.kg.nodes_of_type(NodeType.PRODUCT)
        self.assertEqual(len(prods), 900)
        uncategorised = [p for p in prods if not self.kg.category_path_of(self.kg.node(p)["parent_asin"])]
        self.assertEqual(uncategorised, [])

    def test_schema_valid(self):
        self.assertEqual(self.kg.validate(), [])

    def test_no_placeholder_brand_nodes(self):
        names = {self.kg.node(b)["name"].casefold() for b in self.kg.nodes_of_type(NodeType.BRAND)}
        self.assertFalse(names & nz.PLACEHOLDER_BRANDS)
        self.assertNotIn("visit the funko store", names)

    def test_reviews_trace_to_source_records(self):
        """Every Review node corresponds to a real review_samples entry of
        the product it is ABOUT, and every non-empty source review is
        represented (exact duplicates merged, never fabricated)."""
        import json
        from knowledge_graph.ontology import review_id
        expected = {}
        with SAMPLE.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                for r in rec.get("review_samples") or []:
                    if nz.clean_text(r.get("text")):
                        rid = review_id(rec["parent_asin"], r)
                        expected.setdefault(rid, [rec["parent_asin"], 0])[1] += 1
        reviews = self.kg.nodes_of_type(NodeType.REVIEW)
        self.assertEqual(set(reviews), set(expected))
        for rid in reviews:
            asin, count = expected[rid]
            node = self.kg.node(rid)
            self.assertEqual(node["parent_asin"], asin)
            self.assertEqual(node["duplicate_count"], count)
            about = [(d, k) for _, d, k in self.kg.g.out_edges(rid, keys=True)]
            self.assertEqual(about, [(f"product:{asin}", "ABOUT")])

    def test_books_brands_are_authors(self):
        b = self.kg.brand_of("1932225323")
        self.assertEqual((b["name"], b["kinds"]), ("David Barton", ["author"]))

    def test_every_edge_has_provenance(self):
        for _, _, d in self.kg.g.edges(data=True):
            self.assertIn("source", d)
            self.assertTrue(d["source_fields"])
            self.assertEqual(d["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
