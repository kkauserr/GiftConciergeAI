"""Integration tests for the public get_evidence() API and its contract.

Independent of the LLM component. Run from CODE/:
    python -m unittest discover -s knowledge_graph/tests -t . -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_graph import api
from knowledge_graph.api import CONTRACT_VERSION, KGService, load_kg
from knowledge_graph.graph import KnowledgeGraph, iter_jsonl
from knowledge_graph.vocabulary import load_interests

SAMPLE = Path(__file__).resolve().parents[3] / "DATA" / "samples" / "sample_corpus_v2.jsonl"
SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "contract" / "get_evidence.schema.json")
                    .read_text(encoding="utf-8"))
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def check_schema(value, schema, path="$", root=SCHEMA):
    """Minimal JSON Schema checker (type, enum, required, properties,
    items, $ref) -- enough to keep the code and the published schema in
    sync without adding a dependency. Returns a list of errors."""
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].split("/")[-1]]
    errors = []
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]

        def ok(t):
            if t == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if t == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            return isinstance(value, _TYPES[t])
        if not any(ok(t) for t in types):
            return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}: missing {k}")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errors += check_schema(value[k], sub, f"{path}.{k}", root)
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += check_schema(v, schema["items"], f"{path}[{i}]", root)
    return errors


@unittest.skipUnless(SAMPLE.exists(), "development sample not found")
class TestGetEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = KGService.from_jsonl(SAMPLE)
        cls.kg = cls.svc.kg
        asins = [cls.kg.g.nodes[p]["parent_asin"] for p in cls.kg.nodes_of_type(api.NodeType.PRODUCT)]
        cls.all = cls.svc.get_evidence(asins, max_reviews=10)
        cls.by_asin = {r["parent_asin"]: r for r in cls.all["results"]}

        def first(pred):
            return next(r["parent_asin"] for r in cls.all["results"] if pred(r))
        cls.with_quality = first(lambda r: r["review_qualities"])
        cls.with_concern = first(lambda r: r["review_concerns"])
        cls.observations_only = first(lambda r: r["review_observations"] and not r["review_qualities"]
                                      and not r["review_concerns"])
        cls.no_interest = first(lambda r: not r["interests"])
        cls.with_interest = first(lambda r: r["interests"])

    # -------------------------------------------------------------- shape
    def test_whole_sample_matches_schema_and_serializes(self):
        self.assertEqual(check_schema(self.all, SCHEMA), [])
        text = json.dumps(self.all, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(text), self.all)

    def test_schema_checker_catches_violations(self):
        import copy
        bad = copy.deepcopy(self.svc.get_evidence([self.with_quality]))
        bad["results"][0]["review_qualities"][0]["polarity"] = "negative"
        del bad["results"][0]["grounding"]
        bad["results"][0]["interests"].append({"label": 3})
        errors = check_schema(bad, SCHEMA)
        self.assertTrue(any("polarity" in e for e in errors))
        self.assertTrue(any("missing grounding" in e for e in errors))
        self.assertTrue(any("interests" in e for e in errors))

    def test_contract_header(self):
        r = self.svc.get_evidence([self.with_interest])
        self.assertEqual(r["contract_version"], CONTRACT_VERSION)
        self.assertEqual(r["vocabulary"], {"interests": "interests@1.1.0", "aspects": "aspects@1.2.0"})

    # -------------------------------------------------------------- inputs
    def test_multiple_asins_preserve_order_and_duplicates(self):
        asins = [self.with_concern, self.no_interest, self.with_quality, self.with_concern]
        r = self.svc.get_evidence(asins)
        self.assertEqual([x["parent_asin"] for x in r["results"]], asins)
        self.assertEqual([x["index"] for x in r["results"]], [0, 1, 2, 3])
        self.assertEqual(r["results"][0], dict(r["results"][3], index=0))

    def test_unknown_asins(self):
        r = self.svc.get_evidence(["NOT_AN_ASIN", self.with_interest, "", None])
        res = r["results"]
        self.assertEqual([x["found"] for x in res], [False, True, False, False])
        self.assertEqual(res[0]["parent_asin"], "NOT_AN_ASIN")
        self.assertEqual(res[0]["interests"], [])
        self.assertIsNone(res[0]["product"])
        codes = [(w["code"], w.get("index")) for w in r["warnings"]]
        self.assertEqual(codes, [("unknown_asin", 0), ("invalid_asin", 2), ("invalid_asin", 3)])
        self.assertEqual(check_schema(r, SCHEMA), [])

    def test_empty_input(self):
        r = self.svc.get_evidence([])
        self.assertEqual((r["results"], r["warnings"]), ([], []))
        self.assertEqual(check_schema(r, SCHEMA), [])

    def test_bad_argument_types(self):
        with self.assertRaises(TypeError):
            self.svc.get_evidence("B0164RRP14")
        with self.assertRaises(TypeError):
            self.svc.get_evidence([self.with_interest], query_interests="dogs")
        for bad in (-1, 1.5, True, "3"):
            with self.assertRaises(ValueError):
                self.svc.get_evidence([self.with_interest], max_reviews=bad)

    def test_valid_and_invalid_interest_labels(self):
        label = self.by_asin[self.with_interest]["interests"][0]["label"]
        r = self.svc.get_evidence([self.with_interest, self.no_interest],
                                  query_interests=[label, "Not A Label", label.replace("_", " ").title(), 7])
        self.assertEqual(r["query"]["query_interests"], [label])
        codes = [w["code"] for w in r["warnings"]]
        self.assertEqual(codes.count("unknown_interest"), 2)
        self.assertIn("normalized_interest", codes)
        hit, miss = r["results"]
        self.assertEqual(hit["query_interest_match"], {"requested": [label], "matched": [label], "unmatched": []})
        self.assertTrue(next(i for i in hit["interests"] if i["label"] == label)["matches_query"])
        self.assertEqual(miss["query_interest_match"]["unmatched"], [label])
        self.assertTrue(miss["found"])  # query interests never filter products

    def test_no_query_interests(self):
        r = self.svc.get_evidence([self.with_interest])
        self.assertIsNone(r["results"][0]["query_interest_match"])
        self.assertFalse(any(i["matches_query"] for i in r["results"][0]["interests"]))

    def test_accepted_labels_equal_vocabulary(self):
        self.assertEqual(self.svc.interest_labels(), sorted(load_interests().labels()))
        self.assertEqual(len(self.svc.interest_vocabulary()), 31)

    # ------------------------------------------------------ product kinds
    def test_product_without_interest_matches(self):
        x = self.svc.get_evidence([self.no_interest])["results"][0]
        self.assertTrue(x["found"])
        self.assertEqual(x["interests"], [])
        self.assertIsNotNone(x["product"]["title"])
        self.assertIsNotNone(x["category"])

    def test_positive_review_evidence(self):
        x = self.svc.get_evidence([self.with_quality], max_reviews=1)["results"][0]
        q = x["review_qualities"][0]
        self.assertEqual((q["conclusion"], q["polarity"]), ("supported", "positive"))
        self.assertGreaterEqual(q["counts"]["positive"], 2)
        self.assertLessEqual(len(q["supporting_reviews"]), 1)
        self.assertEqual(q["supporting_reviews_omitted"], q["counts"]["positive"] - len(q["supporting_reviews"]))
        path = next(p for p in x["reasoning_paths"] if p["id"] == q["path_id"])
        self.assertTrue(path["assertable"])
        self.assertEqual(path["edges"][0]["rel"], "HAS_ASPECT")
        self.assertIn(q["path_id"], [c["path_id"] for c in x["grounding"]["assertable_claims"]])

    def test_negative_concerns(self):
        x = self.svc.get_evidence([self.with_concern])["results"][0]
        c = x["review_concerns"][0]
        self.assertEqual((c["status"], c["polarity"]), ("concern", "negative"))
        self.assertGreater(c["counts"]["negative"], c["counts"]["positive"])
        self.assertTrue(c["examples"])
        self.assertNotIn(c["label"], [q["label"] for q in x["review_qualities"]])
        self.assertIn(c["path_id"], [k["path_id"] for k in x["grounding"]["cautions"]])
        self.assertNotIn(c["path_id"], [k["path_id"] for k in x["grounding"]["assertable_claims"]])

    def test_only_individual_observations(self):
        x = self.svc.get_evidence([self.observations_only])["results"][0]
        self.assertEqual((x["review_qualities"], x["review_concerns"]), ([], []))
        self.assertTrue(x["review_observations"])
        for o in x["review_observations"]:
            path = next(p for p in x["reasoning_paths"] if p["id"] == o["path_id"])
            self.assertFalse(path["assertable"])
        self.assertFalse(any(c["basis"] == "review_aggregate" for c in x["grounding"]["assertable_claims"]))

    def test_max_reviews_limits_snippets(self):
        x0 = self.svc.get_evidence([self.with_quality], max_reviews=0)["results"][0]
        self.assertEqual(x0["review_observations"], [])
        self.assertEqual(x0["review_qualities"][0]["supporting_reviews"], [])
        self.assertTrue(x0["review_summary"]["observations_omitted"])
        r = self.svc.get_evidence([self.with_quality], max_reviews=50)
        self.assertEqual(r["query"]["max_reviews"], api.MAX_REVIEWS_LIMIT)
        self.assertEqual(r["warnings"][0]["code"], "max_reviews_capped")
        for res in self.all["results"]:
            counts = {}
            for o in res["review_observations"]:
                counts[o["aspect"]] = counts.get(o["aspect"], 0) + 1
                for s in o["snippets"]:
                    self.assertLessEqual(len(s["text"]), api.MAX_SNIPPET_CHARS)
            self.assertTrue(all(n <= 10 for n in counts.values()))

    # ----------------------------------------------- safety invariants
    def test_negative_or_mixed_never_becomes_positive(self):
        for res in self.all["results"]:
            obs = {(o["review_id"], o["aspect"]): o for o in res["review_observations"]}
            concern_labels = {c["label"] for c in res["review_concerns"]}
            for q in res["review_qualities"]:
                self.assertNotIn(q["label"], concern_labels)
                for s in q["supporting_reviews"]:
                    e = self.kg.edge(s["review_id"], api.RelType.SUPPORTS, f"aspect:{q['label']}")
                    self.assertEqual(e["polarity"], "positive")
                    self.assertFalse(e["rating_conflict"])
                    if (s["review_id"], q["label"]) in obs:
                        self.assertEqual(obs[(s["review_id"], q["label"])]["polarity"], "positive")
            for c in res["review_concerns"]:
                for ex in c["examples"]:
                    self.assertEqual(obs[(ex["review_id"], c["label"])]["polarity"], "negative")
            kinds = {p["id"]: p["kind"] for p in res["reasoning_paths"]}
            for claim in res["grounding"]["assertable_claims"]:
                self.assertNotIn(kinds[claim["path_id"]], ("review_concern", "review_observation"))

    def test_reasoning_paths_resolve_to_graph_edges(self):
        for res in self.all["results"][:200]:
            ids = [p["id"] for p in res["reasoning_paths"]]
            self.assertEqual(len(ids), len(set(ids)))
            for p in res["reasoning_paths"]:
                for e in p["edges"]:
                    self.assertTrue(self.kg.g.has_edge(e["src"], e["dst"], key=e["rel"]), e)

    def test_api_does_not_mutate_graph(self):
        before = (self.kg.g.number_of_nodes(), self.kg.g.number_of_edges())
        self.svc.get_evidence([self.with_quality, "NOPE"], ["dogs"])
        self.assertEqual((self.kg.g.number_of_nodes(), self.kg.g.number_of_edges()), before)
        self.assertEqual(self.kg.validate(), [])


class TestIntegrationDoc(unittest.TestCase):
    """INTEGRATION.md must stay in sync with the vocabulary and the schema."""
    DOC = (Path(__file__).resolve().parents[1] / "INTEGRATION.md").read_text(encoding="utf-8")

    def test_documented_labels_equal_vocabulary(self):
        import re
        section = self.DOC.split("### Interest labels", 1)[1].split("How labels are matched", 1)[0]
        documented = re.findall(r"^\| `([a-z_]+)` \|", section, re.MULTILINE)
        self.assertEqual(documented, load_interests().labels())

    def test_documented_versions(self):
        v = load_interests().version_tag
        self.assertIn(v, self.DOC)
        self.assertIn(CONTRACT_VERSION, self.DOC)

    def test_example_result_matches_schema(self):
        import re
        blocks = re.findall(r"```json\n(.*?)```", self.DOC, re.DOTALL)
        example = json.loads(next(b for b in blocks if b.lstrip().startswith('{\n  "index"')))
        self.assertEqual(check_schema(example, SCHEMA["$defs"]["result"]), [])


class TestLoading(unittest.TestCase):
    def test_repeated_calls_do_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "small.jsonl"
            with open(SAMPLE, encoding="utf-8") as src, open(small, "w", encoding="utf-8") as dst:
                for i, line in enumerate(src):
                    if i >= 30:
                        break
                    dst.write(line)
            asin = json.loads(small.read_text(encoding="utf-8").splitlines()[0])["parent_asin"]
            with mock.patch.object(KnowledgeGraph, "build", wraps=KnowledgeGraph.build) as build:
                a = load_kg(small)
                r1 = a.get_evidence([asin])
                b = load_kg(small)
                r2 = b.get_evidence([asin])
                self.assertIs(a, b)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(r1, r2)
                load_kg(small, reload=True)
                self.assertEqual(build.call_count, 2)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = KGService.from_jsonl(SAMPLE, limit=40)
            svc.kg.export_jsonl(Path(tmp) / "kg")
            with mock.patch.object(KnowledgeGraph, "build") as build:
                loaded = KGService.from_export(Path(tmp) / "kg")
                build.assert_not_called()
            asins = [svc.kg.g.nodes[p]["parent_asin"] for p in svc.kg.nodes_of_type(api.NodeType.PRODUCT)]
            self.assertEqual(loaded.get_evidence(asins, ["dogs"]), svc.get_evidence(asins, ["dogs"]))
            self.assertEqual(loaded.kg.validate(), [])

    def test_export_without_meta_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "nodes.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                KGService.from_export(tmp)

    def test_graph_without_vocabulary_is_rejected(self):
        kg = KnowledgeGraph.build(iter_jsonl(SAMPLE, 5), inference=False)
        with self.assertRaises(ValueError):
            KGService.from_graph(kg)

    def test_load_kg_argument_check(self):
        with self.assertRaises(ValueError):
            load_kg(path=SAMPLE, export_dir="x")


if __name__ == "__main__":
    unittest.main()
