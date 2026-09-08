"""Small deterministic tests for scientifically material evaluator conventions."""
import unittest
import contextlib
import io

import numpy as np

import inspec_evaluation as ev
from e1_baseline_extraction import (check_reference_functions, summary, extract_tfidf,
                                    extract_keybert, VECTORIZER_CONFIG, evaluate)
import pandas as pd


class InspecSemantics(unittest.TestCase):
    def test_reference_source_matches(self):
        self.assertTrue(check_reference_functions()["ast_identical"])

    def test_cleaning_preserves_first_occurrence_order(self):
        raw = '[" Beta! ", "alpha-beta", "beta", "NULL", "", "Café", "x_y"]'
        self.assertEqual(ev.clean_list(ev.safe_parse_list(raw)),
                         ["beta", "alpha beta", "caf", "x y"])
        self.assertEqual(ev.safe_parse_list("not a list, two"), ["not a list", "two"])
        self.assertEqual(ev.safe_parse_list(float("nan")), [])
        self.assertEqual(ev.clean_list(["null"], keep_null=True), ["null"])

    def test_empty_conventions(self):
        def no_embedding(_):
            self.fail("Empty lists must not invoke embeddings")
        for gold, pred, expected in [([], [], 1.0), ([], ["a"], 0.0), (["a"], [], 0.0)]:
            self.assertEqual(ev.jaccard(gold, pred), expected)
            self.assertEqual(ev.global_concat_similarity(gold, pred, no_embedding), expected)
            self.assertTrue(all(v == expected for v in
                                ev.soft_matching_metrics(gold, pred, no_embedding, ev.TAU).values()))

    def test_many_to_one_and_mean_max(self):
        lookup = {"g1": [1., 0.], "g2": [0., 1.], "p1": [1., 0.], "p2": [1., 0.]}
        embed = lambda xs: np.array([lookup[x] for x in xs])
        result = ev.soft_matching_metrics(["g1", "g2"], ["p1", "p2"], embed, ev.TAU)
        self.assertEqual(result["soft_precision"], 1.0)
        self.assertEqual(result["soft_recall"], 0.5)
        self.assertAlmostEqual(result["soft_f1"], 2 / 3)
        self.assertAlmostEqual(result["soft_mean_max"], 0.75)
        result = ev.soft_matching_metrics(["p1", "p2"], ["g1"], embed, ev.TAU)
        self.assertEqual(result["soft_recall"], 1.0)

    def test_threshold_and_lexical_phrases(self):
        embed = lambda xs: np.array([[1., 0.] if x == "g" else [0.8, 0.6] for x in xs])
        low = ev.soft_matching_metrics(["g"], ["p"], embed, 0.70)
        high = ev.soft_matching_metrics(["g"], ["p"], embed, 0.90)
        self.assertEqual(low["soft_f1"], 1.0)
        self.assertEqual(high["soft_f1"], 0.0)
        self.assertAlmostEqual(low["soft_mean_max"], high["soft_mean_max"])
        self.assertEqual(ev.jaccard(["machine learning"], ["machine", "learning"]), 0.0)

    def test_global_concatenation_order(self):
        calls = []
        def embed(xs):
            calls.append(xs)
            return np.array([[1., 0.], [0., 1.]])
        self.assertEqual(ev.global_concat_similarity(["z", "a"], ["b", "a"], embed), 0.0)
        self.assertEqual(calls, [["z ; a", "b ; a"]])

    def test_document_mean_and_sample_sd(self):
        frame = pd.DataFrame([{"method": "test", **{k: value for k in ev.METRICS}}
                              for value in [0., 1.]])
        result = summary(frame)
        self.assertTrue((result["mean"] == 0.5).all())
        np.testing.assert_allclose(result.sd, np.sqrt(0.5))
        self.assertTrue((result.n == 2).all())

    def test_missing_prediction_values_are_scored_without_dropping_rows(self):
        gold = pd.DataFrame({"row_abs": range(2000), "doc_id": [str(i) for i in range(2000)],
                             "keywords_gt": ["['gold']"] * 2000})
        predictions = pd.DataFrame({"row_abs": range(2000), "keywords": [float("nan")] * 2000})
        def no_embedding(_):
            self.fail("Missing predictions must follow the one-empty-list convention")
        with contextlib.redirect_stdout(io.StringIO()):
            scores, _ = evaluate(gold, predictions, no_embedding, "fixture")
        self.assertEqual(len(scores), 2000)
        self.assertTrue((scores[ev.METRICS] == 0).all().all())


class BaselineAdapters(unittest.TestCase):
    def test_tfidf_ties_and_empty_document_are_preserved(self):
        inputs = pd.DataFrame({"row_abs": [0, 1], "doc_id": ["a", "b"],
                               "insumo": ["delta beta gamma alpha epsilon zeta", ""]})
        result, _ = extract_tfidf(inputs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.raw_n.tolist(), [5, 0])
        self.assertEqual(result.status.tolist(), ["ok", "empty_output"])
        self.assertEqual(ev.safe_parse_list(result.iloc[0].keywords),
                         ["alpha", "alpha epsilon", "alpha epsilon zeta", "beta", "beta gamma"])

    def test_keybert_batch_adapter_matches_native_ranking(self):
        from keybert import KeyBERT
        from keybert.backend import BaseEmbedder
        from sklearn.feature_extraction.text import CountVectorizer
        # Deterministic toy embeddings test batching/ranking without loading a model.
        class ToyModel:
            def encode(self, texts, **kwargs):
                return np.array([[len(s), sum(s.encode()) % 101 + 1, s.count(" ") + 1]
                                 for s in texts], dtype=np.float32)
        class NativeToyBackend(BaseEmbedder):
            def embed(self, texts, verbose=False):
                return ToyModel().encode(texts)
        inputs = pd.DataFrame({"row_abs": [0, 1], "doc_id": ["a", "b"],
                               "insumo": ["alpha beta gamma delta epsilon zeta", "neural networks predict scientific concepts"]})
        native = KeyBERT(model=NativeToyBackend()).extract_keywords(
            inputs.insumo.tolist(), vectorizer=CountVectorizer(**VECTORIZER_CONFIG),
            top_n=5, use_mmr=False, use_maxsum=False)
        result, _ = extract_keybert(inputs, ToyModel())
        self.assertEqual([ev.safe_parse_list(s) for s in result.keywords],
                         [[phrase for phrase, _ in pairs] for pairs in native])
        self.assertEqual(result.raw_n.tolist(), [5, 5])


if __name__ == "__main__":
    unittest.main()
