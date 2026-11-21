"""Interest and attribute-aspect inference tests.

Most false-positive cases are reduced from real sample products (titles
shortened). Inline records are test inputs only.

Run from CODE/:  python -m unittest discover -s knowledge_graph/tests -t . -v
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from knowledge_graph.extractors import Extraction
from knowledge_graph.graph import KnowledgeGraph, iter_jsonl
from knowledge_graph.inference import make_aspect_extractor, make_interest_extractor
from knowledge_graph.ontology import (
    ASPECT_QUALITY, Edge, Node, NodeType, Provenance, RelType, aspect_id, interest_id,
    product_id,
)
from knowledge_graph.reports import coverage, select_for_labeling, write_labeling_sheet
from knowledge_graph.vocabulary import (
    VocabularyError, PhraseMatcher, load_aspects, load_interests,
)

SAMPLE = Path(__file__).resolve().parents[3] / "DATA" / "samples" / "sample_corpus_v2.jsonl"
INTERESTS = load_interests()
ASPECTS = load_aspects()
EXPECTED_INTERESTS = [
    "history", "mystery_thriller", "scifi_fantasy", "romance", "poetry_literature",
    "biography_memoir", "horror_spooky", "humor", "faith_spirituality", "science_tech",
    "space_astronomy", "dinosaurs", "animals_wildlife", "dogs", "cats", "cooking_baking",
    "coffee_tea", "gardening_plants", "outdoors_camping", "fishing_hunting", "sports_fitness",
    "travel", "music", "art_drawing", "photography", "crafts_diy", "tabletop_games",
    "puzzles", "vehicles_cars", "superheroes_comics", "anime_manga",
]
EXPECTED_ATTRIBUTE_ASPECTS = [
    "handcrafted", "personalizable", "material_wood", "material_precious_metal",
    "material_metal", "material_ceramic_glass", "material_leather", "material_textile",
    "material_plastic_rubber", "material_paper", "multiplayer", "battery_powered",
    "assembly_required", "educational_focus", "collectible", "illustrated",
]


def rec(domain="Toys & Games", title="Plain item", categories=None, features=None,
        details=None, asin="B0TEST0001", **kw):
    r = {"parent_asin": asin, "domain": domain, "title": title,
         "categories": categories if categories is not None else [domain],
         "features": features if features is not None else [],
         "details": details if details is not None else {}, "brand_or_author": "Acme"}
    r.update(kw)
    return r


def interests(record):
    ex = make_interest_extractor(INTERESTS)(record)
    return {e.dst.split(":", 1)[1] for e in ex.edges}


def interest_edges(record):
    return make_interest_extractor(INTERESTS)(record).edges


def aspects(record):
    ex = make_aspect_extractor(ASPECTS)(record)
    return {e.dst.split(":", 1)[1] for e in ex.edges}


def build(records):
    return KnowledgeGraph.build(records)


# ---------------------------------------------------------------- vocabulary
class TestVocabulary(unittest.TestCase):
    def test_labels_and_counts(self):
        self.assertEqual(INTERESTS.labels(), EXPECTED_INTERESTS)
        self.assertEqual(len(INTERESTS.labels()), 31)
        self.assertEqual(ASPECTS.labels(), EXPECTED_ATTRIBUTE_ASPECTS)
        self.assertEqual(len(ASPECTS.reserved), 12)

    def test_versions(self):
        self.assertEqual(INTERESTS.version_tag, "interests@1.1.0")
        self.assertEqual(ASPECTS.version_tag, "aspects@1.2.0")

    def test_quality_aspects_have_no_rules(self):
        for c in ASPECTS.reserved:
            self.assertEqual(c.aspect_class, ASPECT_QUALITY)
            self.assertEqual(c.rules, [])

    def test_occult_not_mapped(self):
        for c in INTERESTS.concepts:
            for r in c.rules:
                self.assertNotIn("occult & paranormal", [s.segment.casefold() for s in r.segments])

    def test_photography_separate_from_art(self):
        self.assertIn("photography", INTERESTS.labels())
        self.assertIn("art_drawing", INTERESTS.labels())

    def _write(self, data):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        return Path(f.name)

    def test_rejects_recipient_label(self):
        data = json.loads(Path(INTERESTS_PATH()).read_text(encoding="utf-8"))
        data["interests"].append({"label": "mom", "category": ["Moms"]})
        with self.assertRaises(VocabularyError):
            load_interests(self._write(data))

    def test_rejects_quality_aspect_rules(self):
        data = json.loads(Path(ASPECTS_PATH()).read_text(encoding="utf-8"))
        data["quality_aspects"][0]["phrases"] = ["durable"]
        with self.assertRaises(VocabularyError):
            load_aspects(self._write(data))

    def test_rejects_duplicate_label(self):
        data = json.loads(Path(INTERESTS_PATH()).read_text(encoding="utf-8"))
        data["interests"].append(dict(data["interests"][0]))
        with self.assertRaises(VocabularyError):
            load_interests(self._write(data))


def INTERESTS_PATH():
    from knowledge_graph.vocabulary import INTERESTS_PATH as p
    return p


def ASPECTS_PATH():
    from knowledge_graph.vocabulary import ASPECTS_PATH as p
    return p


# ------------------------------------------------------------ phrase matcher
class TestPhraseMatcher(unittest.TestCase):
    def test_whole_words_only(self):
        m = PhraseMatcher(["cat"])
        self.assertIsNone(m.find("category scatter catalog"))
        self.assertIsNotNone(m.find("Black Cat Earrings"))

    def test_exclusion_overlap(self):
        m = PhraseMatcher(["dog"], ["hot dog"])
        self.assertIsNone(m.find("Hot Dog costume"))
        self.assertEqual(m.find("Hot dog and a dog").start, 14)

    def test_space_hyphen_equivalence(self):
        m = PhraseMatcher(["t rex"])
        self.assertIsNotNone(m.find("T-Rex figure"))

    def test_curly_apostrophe_exclusion(self):
        m = PhraseMatcher(["cat"], ["cat's eye"])
        self.assertIsNone(m.find("Pink Cat’s Eye Quartz"))

    def test_negation_same_clause(self):
        m = PhraseMatcher(["wood"])
        self.assertIsNone(m.find("It's not printed on the wood, stone or slate"))
        self.assertIsNotNone(m.find("No assembly needed. Solid wood blocks"))

    def test_keyword_list_guard(self):
        m = PhraseMatcher(["cat lover"])
        text = "Gift for boyfriend, him, kids, teens, best friend, cat lover, Christmas, Xmas"
        self.assertIsNotNone(m.find(text))
        self.assertIsNone(m.find(text, guard_keyword_lists=True))

    def test_long_compatible_uses_list_guard(self):
        m = PhraseMatcher(["guitars"])
        text = "Stick on laptops, iPads, notebooks, toolboxes, guitars, luggage, refrigerators, window, car"
        self.assertIsNone(m.find(text, guard_keyword_lists=True))

    def test_real_sentence_survives_guard(self):
        m = PhraseMatcher(["guitar"])
        self.assertIsNotNone(m.find("Guitar music decal, vinyl sticker", guard_keyword_lists=True))


# ----------------------------------------------------- interest false positives
class TestInterestFalsePositives(unittest.TestCase):
    def test_books_title_word_senses(self):
        thr = ["Books", "Mystery, Thriller & Suspense", "Thrillers & Suspense"]
        self.assertEqual(interests(rec("Books", "Hunting Evil", thr)), {"mystery_thriller"})
        lit = ["Books", "Literature & Fiction", "Essays"]
        self.assertEqual(interests(rec("Books", "Out of Sheer Rage: Wrestling with D. H. Lawrence", lit)), set())
        self.assertEqual(interests(rec("Books", "Full Moon: The Amazing Rock and Roll Life", lit)), set())

    def test_books_features_never_used(self):
        r = rec("Books", "A Novel", ["Books", "Literature & Fiction"],
                features=["She travels through time and space with her dog to a secret garden, fishing."])
        self.assertEqual(interests(r), set())

    def test_cats_eye_quartz(self):
        self.assertNotIn("cats", interests(rec("Handmade Products", "Pink Cat's Eye Quartz Ring")))

    def test_marvel_verb(self):
        self.assertNotIn("superheroes_comics",
                         interests(rec("Handmade Products", "Earrings - Marvel at the depths of blue")))

    def test_toy_kitchen_not_coffee_or_cooking(self):
        path = ["Toys & Games", "Preschool", "Pre-Kindergarten Toys", "Pretend Play", "Kitchen Sets & Play Food"]
        self.assertEqual(interests(rec(title="New Classic Toys Coffee Maker, Baking Set", categories=path)), set())
        self.assertIn("coffee_tea", interests(rec("Handmade Products", "Handmade Coffee Lover Mug")))

    def test_recipient_occasion_keyword_list(self):
        r = rec("Handmade Products", "Phone Grip",
                features=["Gift for girlfriend, boyfriend, him, kids, teens, best friend, cat lover, Christmas, Xmas"])
        self.assertEqual(interests(r), set())

    def test_recipient_and_occasion_alone_give_nothing(self):
        self.assertEqual(interests(rec("Handmade Products", "Gift for Mom Birthday Christmas Anniversary Teacher Nurse")), set())

    def test_occult_not_mapped(self):
        r = rec("Books", "Ghosts of the Midwest", ["Books", "Religion & Spirituality", "Occult & Paranormal"])
        self.assertEqual(interests(r), set())
        self.assertEqual(interests(rec("Books", "Faith", ["Books", "Religion & Spirituality"])),
                         {"faith_spirituality"})

    def test_company_name_dark_horse(self):
        self.assertNotIn("animals_wildlife", interests(rec(title="Dark Horse Deluxe Game of Thrones Figure")))

    def test_crocheted_product_is_not_crafter_interest(self):
        self.assertNotIn("crafts_diy", interests(rec("Handmade Products", "Crochet Sunflower Garland")))
        self.assertIn("crafts_diy", interests(rec("Handmade Products", "Ergonomic Crochet Hook Set")))

    def test_costume_occasion_suppression(self):
        path = ["Toys & Games", "Preschool", "Pre-Kindergarten Toys", "Pretend Play", "Role Play & Dress Up"]
        self.assertNotIn("horror_spooky", interests(rec(title="Zombie Deluxe Costume for Child", categories=path)))

    def test_car_decal_not_vehicle_interest(self):
        path = ["Handmade Products", "Sports & Outdoors", "Car & Vehicle Accessories", "Decals"]
        self.assertNotIn("vehicles_cars", interests(rec("Handmade Products", "Monster Truck Decal", path)))

    def test_keychains_under_travel_gear_are_not_travel(self):
        kc = ["Handmade Products", "Clothing, Shoes & Accessories", "Luggage & Travel Gear",
              "Key & Identification Accessories", "Keychains & Keyrings"]
        self.assertEqual(interests(rec("Handmade Products", "Initial Keychain", kc)), set())
        pc = ["Handmade Products", "Clothing, Shoes & Accessories", "Luggage & Travel Gear",
              "Travel Accessories", "Passport Covers & Wallets"]
        self.assertEqual(interests(rec("Handmade Products", "Leather Cover", pc)), {"travel"})

    def test_yard_sign_not_gardening(self):
        path = ["Handmade Products", "Home & Kitchen", "Patio, Lawn & Garden", "Outdoor Décor", "Yard Signs"]
        self.assertNotIn("gardening_plants", interests(rec("Handmade Products", "Welcome Sign", path)))

    def test_rc_dog_bone_part(self):
        path = ["Toys & Games", "Remote & App Controlled Vehicles & Parts", "Remote & App Controlled Vehicle Parts"]
        found = interests(rec(title="Losi Dog Bone Set Mini 8IGHT", categories=path))
        self.assertNotIn("dogs", found)
        self.assertIn("vehicles_cars", found)


class TestInterestPositives(unittest.TestCase):
    def test_books_category(self):
        (e,) = interest_edges(rec("Books", "Four Centuries", ["Books", "History", "Americas"]))
        self.assertEqual((e.dst, e.provenance.source, e.provenance.confidence),
                         (interest_id("history"), "category_map", 0.9))
        self.assertEqual(e.provenance.rule_id, "history.category")
        self.assertEqual(e.attrs["vocab_version"], "interests@1.1.0")
        self.assertEqual(e.attrs["match_type"], "primary")

    def test_handmade_title_with_span(self):
        r = rec("Handmade Products", "Personalized Fishing Gifts - Bass Fish Ornament")
        (e,) = [x for x in interest_edges(r) if x.dst == interest_id("fishing_hunting")]
        self.assertEqual(e.provenance.source, "title_lexicon")
        s, t = e.provenance.span
        self.assertEqual(r["title"][s:t], e.provenance.evidence)

    def test_theme_detail(self):
        r = rec(details={"Theme": "Animal"})
        (e,) = interest_edges(r)
        self.assertEqual((e.dst, e.provenance.source_fields),
                         (interest_id("animals_wildlife"), ("details.Theme",)))
        self.assertEqual(e.attrs["match_type"], "themed")

    def test_photography_category(self):
        path = ["Handmade Products", "Electronics Accessories", "Camera & Photo Accessories"]
        self.assertEqual(interests(rec("Handmade Products", "Strap", path)), {"photography"})


# ------------------------------------------------------- aggregation / graph
class TestAggregation(unittest.TestCase):
    def test_max_not_noisy_or(self):
        path = ["Toys & Games", "Puzzles", "Jigsaw Puzzles"]
        r = rec(title="1000 Piece Jigsaw Puzzle", categories=path,
                features=["A challenging jigsaw for the whole family"])
        kg = build([r])
        e = kg.edge(product_id(r["parent_asin"]), RelType.MATCHES, interest_id("puzzles"))
        self.assertEqual(e["confidence"], 0.9)  # max of 0.9/0.7/0.5, not combined
        self.assertEqual(sorted(x["source"] for x in e["evidence_list"]),
                         ["category_map", "feature_lexicon", "title_lexicon"])
        self.assertEqual(e["source"], "category_map")  # top level mirrors the best item

    def test_distinct_evidence_not_duplicated(self):
        r = rec("Handmade Products", "Dog Mom Mug")
        kg = build([r, dict(r)])  # same record twice
        e = kg.edge(product_id(r["parent_asin"]), RelType.MATCHES, interest_id("dogs"))
        self.assertEqual(len(e["evidence_list"]), 1)

    def test_feature_only_is_incidental(self):
        # v1.1: feature text corroborates but never creates a MATCHES edge.
        r = rec(features=["Includes a realistic cat figure"])
        ex = make_interest_extractor(INTERESTS)(r)
        self.assertEqual(ex.edges, [])
        self.assertEqual(ex.flags["interest_incidental_mentions"], 1)
        kg = build([r])
        self.assertEqual(kg.interests_of(r["parent_asin"]), [])
        self.assertEqual(kg.validate(), [])

    def test_vocabulary_nodes_always_present(self):
        kg = build([])
        self.assertEqual(len(kg.nodes_of_type(NodeType.INTEREST)), 31)
        # 16 attribute aspects + 12 review-only quality aspects (SUPPORTS targets)
        self.assertEqual(len(kg.nodes_of_type(NodeType.ASPECT)), 28)


class TestMissingMetadata(unittest.TestCase):
    def test_empty_record_fields(self):
        r = {"parent_asin": "B0EMPTY001", "domain": "Toys & Games", "title": None,
             "categories": None, "features": None, "details": None}
        kg = build([r])
        self.assertEqual(kg.interests_of("B0EMPTY001"), [])
        self.assertEqual(kg.aspects_of("B0EMPTY001"), [])
        self.assertEqual(kg.validate(), [])

    def test_malformed_types(self):
        r = rec(title="Plush Dog", features=["Dog toy", 42, None], details="not a dict",
                categories="Toys")
        self.assertEqual(interests(r), {"dogs"})
        (e,) = interest_edges(r)[:1]
        self.assertEqual(e.provenance.source, "title_lexicon")
        self.assertEqual(aspects(r), set())

    def test_no_match_product_remains_eligible(self):
        r = rec("Handmade Products", "Silver Hoop Earrings",
                ["Handmade Products", "Jewelry", "Earrings", "Hoop"])
        kg = build([r])
        self.assertEqual(kg.interests_of(r["parent_asin"]), [])
        self.assertIn(product_id(r["parent_asin"]),
                      kg.products_in_category("category:handmade-products"))
        self.assertIsNotNone(kg.product_summary(r["parent_asin"]))

    def test_domain_fallback_category_not_used_for_interests(self):
        # Empty categories fall back to the domain root for BELONGS_TO, but that
        # fallback must not count as category evidence for interests.
        r = rec("Toys & Games", "Thing", categories=[])
        self.assertEqual(interests(r), set())


# --------------------------------------------------------------------- aspects
class TestAspects(unittest.TestCase):
    def test_handmade_domain(self):
        ex = make_aspect_extractor(ASPECTS)(rec("Handmade Products"))
        (e,) = ex.edges
        self.assertEqual((e.dst, e.provenance.evidence_type, e.provenance.confidence),
                         (aspect_id("handcrafted"), "structured", 0.9))

    def test_material_details_multi_value(self):
        self.assertEqual(aspects(rec(details={"Material": "Wood, Plastic"})),
                         {"material_wood", "material_plastic_rubber"})

    def test_faux_leather_and_plated_gold(self):
        self.assertEqual(aspects(rec(details={"Material": "Faux Leather"})), set())
        self.assertNotIn("material_precious_metal",
                         aspects(rec("Toys & Games", "14k Gold Plated Charm Bracelet")))
        self.assertIn("material_precious_metal",
                      aspects(rec("Toys & Games", "925 Sterling Silver Charm")))

    def test_plastic_free_and_alternative(self):
        self.assertNotIn("material_plastic_rubber",
                         aspects(rec(features=["Plastic-free packaging, eco friendly"])))
        self.assertNotIn("material_plastic_rubber",
                         aspects(rec(features=["A petroleum-free alternative to plastic dryer balls"])))

    def test_packaging_material_is_not_product_material(self):
        self.assertNotIn("material_textile", aspects(rec(features=["Black velvet gift bag included"])))

    def test_players_batteries_assembly(self):
        self.assertIn("multiplayer", aspects(rec(details={"Number of Players": "2-4"})))
        self.assertNotIn("multiplayer", aspects(rec(details={"Number of Players": "1"})))
        self.assertIn("battery_powered", aspects(rec(details={"Batteries": "3 AA batteries required."})))
        self.assertNotIn("battery_powered", aspects(rec(details={"Batteries Required?": "No"})))
        self.assertIn("assembly_required", aspects(rec(details={"Assembly Required": "Yes"})))
        self.assertNotIn("assembly_required", aspects(rec(details={"Assembly Required": "No"})))

    def test_personalization_wording(self):
        self.assertIn("personalizable", aspects(rec("Handmade Products", "Custom Name Necklace")))
        self.assertNotIn("personalizable", aspects(rec(
            "Books", "Personalized Learning in the Classroom", ["Books", "Education & Teaching"])))
        self.assertNotIn("personalizable", aspects(rec(
            "Handmade Products", "Magnet", features=["Laser cut, engraved and hand assembled in NJ"])))

    def test_books_no_material_from_text(self):
        r = rec("Books", "Leather-bound Wooden Boat Building Guide", ["Books", "Crafts, Hobbies & Home"])
        self.assertEqual({a for a in aspects(r) if a.startswith("material_")}, set())

    def test_books_illustrated_only_explicit_phrases(self):
        lit = ["Books", "Literature & Fiction"]
        self.assertIn("illustrated", aspects(rec("Books", "A Tale", lit, features=["Illustrated by Jane Doe"])))
        self.assertNotIn("illustrated", aspects(rec("Books", "A Tale", lit,
                                                    features=["The war illustrated how fragile peace was"])))

    def test_seller_quality_claims_ignored(self):
        r = rec(features=["Durable, sturdy, premium high quality, safe and non-toxic, educational"])
        self.assertEqual(aspects(r), set())

    def test_validate_rejects_quality_from_listing(self):
        kg = build([rec()])
        q = ASPECTS.get("durability")
        kg.add_node(Node(q.node_id, NodeType.ASPECT, q.node_attrs()))
        kg.add_edge(Edge(product_id("B0TEST0001"), RelType.HAS_ASPECT, q.node_id,
                         Provenance("feature_lexicon", ["features[0]"], 0.6, "durable",
                                    rule_id="durability.features", evidence_type="listing_stated"),
                         {"vocab_version": ASPECTS.version_tag}))
        self.assertTrue(any("quality aspect without review evidence" in p for p in kg.validate()))

    def test_validate_rejects_book_feature_interest(self):
        kg = build([rec("Books", "A Novel", ["Books", "Literature & Fiction"])])
        kg.add_edge(Edge(product_id("B0TEST0001"), RelType.MATCHES, interest_id("dogs"),
                         Provenance("feature_lexicon", ["features[0]"], 0.5, "dog", rule_id="dogs.features"),
                         {"vocab_version": INTERESTS.version_tag}))
        self.assertTrue(any("Book feature text" in p for p in kg.validate()))


# ---------------------------------------------------------------- full sample
@unittest.skipUnless(SAMPLE.exists(), "development sample not found")
class TestSampleInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = KnowledgeGraph.build(iter_jsonl(SAMPLE))

    def test_valid(self):
        self.assertEqual(self.kg.validate(), [])

    def test_all_products_present(self):
        self.assertEqual(len(self.kg.nodes_of_type(NodeType.PRODUCT)), 900)

    def test_inferred_edges_have_provenance(self):
        n = 0
        for _, _, k, d in self.kg.g.edges(keys=True, data=True):
            if k in ("MATCHES", "HAS_ASPECT"):
                n += 1
                self.assertTrue(d["rule_id"] and d["vocab_version"] and d["evidence_list"])
                self.assertLess(d["confidence"], 1.0)
                for item in d["evidence_list"]:
                    self.assertTrue(item["source_fields"])
                    self.assertIsNotNone(item["evidence"])
        self.assertGreater(n, 0)

    def test_quality_aspect_edges_only_from_reviews(self):
        for _, v, k, d in self.kg.g.edges(keys=True, data=True):
            if k != "HAS_ASPECT":
                continue
            sources = {x["source"] for x in d["evidence_list"]}
            if self.kg.g.nodes[v]["aspect_class"] == "quality":
                self.assertEqual(sources, {"review_aggregate"})
            else:
                self.assertNotIn("review_aggregate", sources)

    def test_no_quality_edges_without_review_inference(self):
        kg = KnowledgeGraph.build(iter_jsonl(SAMPLE, 60), review_inference=False)
        for _, v, k in kg.g.edges(keys=True):
            self.assertNotEqual(k, "SUPPORTS")
            if k == "HAS_ASPECT":
                self.assertEqual(kg.g.nodes[v]["aspect_class"], "attribute")

    def test_coverage_per_domain(self):
        cov = coverage(self.kg)
        self.assertEqual(set(cov), {"Books", "Handmade Products", "Toys & Games"})
        for d in cov.values():
            self.assertEqual(d["products"], 300)
            self.assertGreater(d["interests"]["products_covered"], 0)

    def test_labeling_sheet(self):
        chosen = select_for_labeling(self.kg)
        self.assertEqual(len(chosen), 60)
        self.assertEqual(sorted({d for d, _, _ in chosen}), ["Books", "Handmade Products", "Toys & Games"])
        for dom in ("Books", "Handmade Products", "Toys & Games"):
            self.assertEqual(sum(1 for d, _, _ in chosen if d == dom), 20)
        self.assertEqual(chosen, select_for_labeling(self.kg))  # deterministic
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.csv"
            self.assertEqual(write_labeling_sheet(self.kg, path), 60)
            with open(path, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 60)
            self.assertEqual(rows[0]["gold_interests"], "")


if __name__ == "__main__":
    unittest.main()
