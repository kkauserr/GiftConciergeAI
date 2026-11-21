"""Regression tests for the v1.1 interest/aspect false-positive cleanup.

Each case is reduced from a real product in the development sample (ASIN
in the comment, titles and features shortened). Inline records are test
inputs only.

Run from CODE/:  python -m unittest discover -s knowledge_graph/tests -t . -v
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from knowledge_graph import normalize as nz
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.inference import make_aspect_extractor, make_interest_extractor
from knowledge_graph.ontology import (
    Edge, Node, NodeType, Provenance, RelType, interest_id, product_id,
)
from knowledge_graph.reports import write_labeling_sheet
from knowledge_graph.vocabulary import component_excluded, load_aspects, load_interests

INTERESTS = load_interests()
ASPECTS = load_aspects()
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def rec(domain="Toys & Games", title="Plain item", categories=None, features=None,
        details=None, asin="B0TEST0001"):
    return {"parent_asin": asin, "domain": domain, "title": title,
            "categories": categories if categories is not None else [domain],
            "features": features or [], "details": details or {}, "brand_or_author": "Acme"}


def interests(record):
    return {e.dst.split(":", 1)[1] for e in make_interest_extractor(INTERESTS)(record).edges}


def interest_extraction(record):
    return make_interest_extractor(INTERESTS)(record)


def aspects(record):
    return {e.dst.split(":", 1)[1] for e in make_aspect_extractor(ASPECTS)(record).edges}


# ------------------------------------------------- incidental mentions
class TestIncidentalMentions(unittest.TestCase):
    def test_hummingbird_jigsaw_cross_sell(self):
        # B09SHKRDTW: a feature points at a different (cat) puzzle.
        r = rec(title="Sunflower Hummingbird Wooden Jigsaw Puzzle",
                categories=["Toys & Games", "Puzzles", "Jigsaw Puzzles"],
                features=["If You Want To Give Gift For Others, You Can Consider This Cat Puzzle."])
        found = interests(r)
        self.assertNotIn("cats", found)
        self.assertIn("puzzles", found)

    def test_cross_sell_sentence_ignored_for_aspects(self):
        r = rec("Handmade Products", "Name Bracelet",
                features=["Adjustable cord. Check out our leather keychains in the shop."])
        self.assertNotIn("material_leather", aspects(r))

    def test_cross_sell_spans_cover_only_that_sentence(self):
        text = "Solid wood frame. You can also check out our metal frames."
        spans = nz.cross_sell_spans(text)
        self.assertEqual([text[s:e] for s, e in spans], ["You can also check out our metal frames."])

    def test_toy_tea_set_accessory(self):
        # B07HP7MSDN: an anime figure that includes a tiny tea set.
        r = rec(title="Beatless: Lacia Nendoroid Action Figure",
                features=["Includes a tea set to recreate everyday scenes"])
        self.assertNotIn("coffee_tea", interests(r))

    def test_kids_camera_puzzle_games(self):
        # B0BRP6FM87
        r = rec(title="Kids Instant Camera for 3-12 Years Old",
                features=["【Creative DIY & 4 Puzzle Games】 Kids can do more than just snap"])
        self.assertNotIn("puzzles", interests(r))

    def test_doll_passport_is_not_travel(self):
        # B0B2JCPF7D
        r = rec(title="18 Inch Doll Clothes and Accessories 24Pcs",
                features=["This little bag holds the doll's passport and boarding pass"])
        self.assertNotIn("travel", interests(r))

    def test_game_piece_list(self):
        # B00064XFW0 Boooo-opoly
        r = rec(title="Boooo-opoly",
                features=["Player pieces include: cat, bat, pumpkin, candy apple, ghost"])
        self.assertNotIn("cats", interests(r))

    def test_feature_only_counted_as_incidental(self):
        ex = interest_extraction(rec(features=["Includes a realistic cat figure"]))
        self.assertEqual(ex.edges, [])
        self.assertEqual(ex.flags["interest_incidental_mentions"], 1)

    def test_features_still_corroborate(self):
        r = rec("Handmade Products", "Dog Collar Charm",
                features=["Engraved with a paw print for dog lovers"])
        edges = [e for e in interest_extraction(r).edges if e.dst == interest_id("dogs")]
        self.assertEqual(sorted(e.provenance.source for e in edges),
                         ["feature_lexicon", "title_lexicon"])

    def test_validate_rejects_feature_only_matches(self):
        kg = KnowledgeGraph.build([rec()])
        pid = product_id("B0TEST0001")
        kg.add_edge(Edge(pid, RelType.MATCHES, interest_id("cats"),
                         Provenance("feature_lexicon", ["features[0]"], 0.5, "cat",
                                    rule_id="cats.features"),
                         {"vocab_version": INTERESTS.version_tag, "match_type": "themed"}))
        self.assertTrue(any("feature text only" in p for p in kg.validate()))


# ------------------------------------------------- keyword-stuffed listings
class TestKeywordStuffing(unittest.TestCase):
    def test_resin_use_list(self):
        # B0992YSSQD: "Fishing Lures" is one of several uses for the resin.
        r = rec(title="Art Resin - Durable, Glossy, Water Clear, Scratch Resistant - Small Cast "
                      "Jewelry, Fishing Lures, Dolls, Hobby, RC, Crafts ~ Cures in Mins")
        self.assertNotIn("fishing_hunting", interests(r))

    def test_short_list_of_two_still_matches(self):
        r = rec("Handmade Products", "Fishing Lure Earrings, Bass, Silver")
        self.assertIn("fishing_hunting", interests(r))

    def test_compatibility_clause(self):
        # B09NLST9HP: a football card binder that "fits" Yugioh cards.
        r = rec(title="Football Card Binder, Trading Card Album Fit for TCG Yugioh card and "
                      "Sports Cards.(football card binder)")
        self.assertNotIn("anime_manga", interests(r))

    def test_multi_valued_theme_ignored(self):
        # B0BP7H64R8: Theme lists motifs, not the product's subject.
        r = rec(title="Wooden Stacking Blocks", details={"Theme": "Animals,Shapes,Vehicles"})
        self.assertNotIn("animals_wildlife", interests(r))

    def test_two_valued_theme_still_used(self):
        r = rec(title="Puppet", details={"Theme": "Animal,Jungle"})
        self.assertIn("animals_wildlife", interests(r))

    def test_theme_anime_not_mapped(self):
        # B002R3HYEI: seller set Theme=Anime on a Magic: The Gathering card.
        r = rec(title="Magic The Gathering - Graypelt Refuge (214) - Zendikar",
                details={"Theme": "Anime"})
        self.assertNotIn("anime_manga", interests(r))

    def test_yugioh_title_phrase(self):
        r = rec(title="Yu-Gi-Oh! Darkwing Blast Booster Box")
        self.assertIn("anime_manga", interests(r))


# ------------------------------------------------- party supplies / nursery
class TestProductClassSuppression(unittest.TestCase):
    def test_nursery_print(self):
        # B097JNVQWQ: the scene is decoration for a baby's room.
        r = rec("Handmade Products",
                "Personalized Monogram Watercolor Little Boy Fishing with his Dog Nursery or "
                "Little Boys Room Print, Rustic Outdoor Decor")
        ex = interest_extraction(r)
        self.assertEqual(ex.edges, [])
        self.assertTrue(ex.flags.get("interest_text_suppressed=infant_nursery"))

    def test_party_favor_bags(self):
        # B092QQ8HVY
        r = rec(title="Mocoosy 18 Pack Dinosaur Party Favor Bags with Handles - Dinosaur Gift Bags")
        self.assertNotIn("dinosaurs", interests(r))

    def test_party_invitations_and_toppers(self):
        self.assertEqual(interests(rec("Handmade Products",
                                       "Monster Truck Birthday Party Invitations - Set of 20")), set())
        self.assertEqual(interests(rec("Handmade Products",
                                       "Chihuahua Cupcake Toppers, set of 6 birthday dogs")), set())

    def test_balloon_theme_suppressed(self):
        # B08PV994RT: Theme field also ignored for party supplies.
        r = rec(title="Jungle Safari Theme Balloon Garland Arch Kit, Animal Balloons",
                details={"Theme": "Animal"})
        self.assertEqual(interests(r), set())

    def test_category_rules_survive_suppression(self):
        r = rec("Handmade Products", "Dog Birthday Party Decorations",
                ["Handmade Products", "Pet Supplies", "Dogs", "Toys"])
        self.assertEqual(interests(r), {"dogs"})

    def test_party_use_suggestion_not_product_type(self):
        # B09NR73XDH: game coins "for Party Favor" are still game accessories.
        r = rec(title="Fantasy Coins Board Game Accessories Metal Game Coin Toys for Party Favor")
        self.assertIn("tabletop_games", interests(r))

    def test_party_word_in_seo_tail(self):
        # B0B77DR76R: a beach ball whose title later mentions party favors.
        r = rec(title="Spider Hero Inflatable Beach Ball,Superhero Toys Game Ball, Pool Party Favors")
        self.assertIn("superheroes_comics", interests(r))

    def test_baby_shower_is_an_occasion_not_a_class(self):
        # B0BW5XV8YW: "Baby Shower" appears in an occasion list.
        r = rec(title="Piggy Bank for Boys Cute Dinosaur Toys Gift Birthday Christmas Baby Shower")
        self.assertIn("dinosaurs", interests(r))

    def test_books_not_suppressed(self):
        self.assertIsNone(INTERESTS.text_suppression_class(
            "Bambini: The Italian Approach to Infant/Toddler Care", "Books"))

    def test_title_head(self):
        from knowledge_graph.vocabulary import title_head
        self.assertEqual(title_head("Beach Ball, Superhero Toys - Party Favors"), "Beach Ball")


# ------------------------------------------------- match types
class TestMatchType(unittest.TestCase):
    def test_category_is_primary(self):
        r = rec(title="Chess Set", categories=["Toys & Games", "Games", "Board Games"])
        (e,) = [x for x in interest_extraction(r).edges if x.dst == interest_id("tabletop_games")][:1]
        self.assertEqual(e.attrs["match_type"], "primary")

    def test_title_motif_is_themed(self):
        r = rec("Handmade Products", "Sterling Silver Dachshund Necklace")
        (e,) = interest_extraction(r).edges
        self.assertEqual(e.attrs["match_type"], "themed")

    def test_books_title_is_primary(self):
        r = rec("Books", "Oregon Birds: A Folding Pocket Guide", ["Books"])
        (e,) = interest_extraction(r).edges
        self.assertEqual(e.attrs["match_type"], "primary")


# ------------------------------------------------- component materials
class TestComponentMaterials(unittest.TestCase):
    def test_kit_contents_list(self):
        # B0757KH7LJ
        r = rec(title="Ultimate Chemistry Lab STEM Kit",
                features=["1 Bottle Dye Coloring, 1 Plastic Funnel, 1 Wooden Scratch Stick, 1 Plastic Flask"])
        found = aspects(r)
        self.assertNotIn("material_plastic_rubber", found)
        self.assertNotIn("material_wood", found)

    def test_counted_party_favor_contents(self):
        # B0B6C7TD2D
        r = rec(title="88 Pcs Halloween Favors",
                features=["set includes 12 wood pencils 16 notebooks and 12 rulers"])
        self.assertNotIn("material_wood", aspects(r))

    def test_single_counted_item_kept(self):
        r = rec("Handmade Products", "Name Necklace", features=["Includes: 1 Stainless Steel Necklace"])
        self.assertIn("material_metal", aspects(r))

    def test_measurements_and_purity_are_not_counts(self):
        self.assertEqual(nz.contents_list_items(
            "Printed on 110 lb card stock, 5 x 7 inch, 925 sterling silver, 14k gold"), [])
        r = rec("Handmade Products", "Ring", features=["925 Sterling Silver band, 18 inch chain"])
        self.assertIn("material_precious_metal", aspects(r))

    def test_not_included(self):
        # B0BKZWRTBK
        r = rec(title="DIY Reborn Kit", features=["Eyes closed. Cotton is not included."])
        self.assertNotIn("material_textile", aspects(r))

    def test_replaced_parts(self):
        # B09T92NMJR and B08TVY9KB6
        r = rec(title="Front Bumper Mount", features=["Directly replace Original plastic parts."])
        self.assertNotIn("material_plastic_rubber", aspects(r))
        r = rec(title="Chemistry Puzzles",
                features=["plastic atom models are replaced with flat, cardboard puzzle pieces"])
        found = aspects(r)
        self.assertNotIn("material_plastic_rubber", found)
        self.assertIn("material_paper", found)

    def test_component_guard_does_not_block_plain_statements(self):
        self.assertFalse(component_excluded("Made of plastic, not wood", 8, 15))
        r = rec(title="Toy Car", features=["Made of BPA-free plastic"])
        self.assertIn("material_plastic_rubber", aspects(r))

    def test_instructions_and_finish(self):
        r = rec(title="Flower Building Kit", features=["Easily followed by the paper instruction."])
        self.assertNotIn("material_paper", aspects(r))
        r = rec("Handmade Products", "Bathroom Sign", features=["See Video for Beautiful Acrylic Finish!"])
        self.assertNotIn("material_plastic_rubber", aspects(r))


# ------------------------------------------------- labeling sheet safety
class TestLabelingSheetPreserved(unittest.TestCase):
    def test_writer_refuses_to_overwrite(self):
        kg = KnowledgeGraph.build([rec()])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.csv"
            path.write_text("annotated", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_labeling_sheet(kg, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "annotated")

    def test_committed_sheet_has_60_rows(self):
        sheet = EVAL_DIR / "labeling_sheet_v1.csv"
        if not sheet.exists():
            self.skipTest("labeling sheet not present")
        import csv
        with open(sheet, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 60)


if __name__ == "__main__":
    unittest.main()
