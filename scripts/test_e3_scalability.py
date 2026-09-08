"""E3 semantic and sampling checks; no model loading or external calls."""
import unittest
import numpy as np
import networkx as nx
import crs_reference as ref
from e3_scalability import sampled_positions, component_stats, reference_audit, metrics


class CRSChecks(unittest.TestCase):
    def test_reference_functions(self):
        self.assertTrue(all(c["ast_identical"] for c in reference_audit().values()))

    def test_support_count_and_local_semantic_filter(self):
        embeddings = {"a": np.array([1., 0.]), "b": np.array([0.8, 0.6]),
                      "c": np.array([0., 1.]), "isolated": np.array([-1., 0.])}
        docs = [["a", "b"], ["a", "b"], ["a", "c"], ["isolated"]]
        graph = ref.build_crs_for_tau(docs, embeddings, 0.4)
        self.assertEqual(graph["a"]["b"]["weight"], 2)
        self.assertAlmostEqual(graph["a"]["b"]["sim_mean"], 0.8)
        self.assertEqual(graph.nodes["a"]["doc_freq"], 3)
        self.assertFalse(graph.has_edge("a", "c"))
        self.assertFalse(graph.has_edge("b", "c"))  # similar, but no shared document
        self.assertIn("isolated", graph)

    def test_absolute_backbone_and_isolates(self):
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=19)
        graph.add_edge("b", "c", weight=20)
        graph.add_node("isolated")
        backbone = ref.build_backbone(graph, 20)
        self.assertEqual(set(backbone), {"b", "c"})
        self.assertEqual(backbone.number_of_edges(), 1)
        empty = ref.build_backbone(graph, 21)
        self.assertEqual(component_stats(empty), (0, 0, 0, 0.0))

    def test_empty_backbone_is_reported_without_adjustment(self):
        graph = ref.build_crs_for_tau([["a", "b"]], {"a": np.array([1., 0.]), "b": np.array([1., 0.])}, 0.4)
        result = metrics(graph)
        self.assertEqual(result["backbone_nodes"], 0)
        self.assertEqual(result["n_communities"], 0)
        self.assertTrue(np.isnan(result["modularity"]))
        self.assertEqual(result["lcc_fraction_nodes"], 1.0)

    def test_sampling_reproducible_unique_not_nested(self):
        for seed in range(42, 52):
            a = sampled_positions(52946, 500, seed)
            b = sampled_positions(52946, 1000, seed)
            self.assertEqual(len(a), 500)
            self.assertEqual(len(np.unique(a)), 500)
            np.testing.assert_array_equal(a, sampled_positions(52946, 500, seed))
            self.assertFalse(set(a).issubset(set(b)))

    def test_reference_parsing_and_normalization(self):
        values = ref.parse_keywords('["  Alpha ", "alpha", "BETA", "null", "a-b", 3, ""]')
        cleaned = sorted(set(k.strip().lower() for k in values if isinstance(k, str) and k.strip()))
        self.assertEqual(cleaned, ["a-b", "alpha", "beta", "null"])
        self.assertEqual(ref.parse_keywords("malformed"), [])
        self.assertEqual(ref.parse_keywords("[]"), [])


if __name__ == "__main__":
    unittest.main()
