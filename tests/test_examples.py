"""Unit + smoke tests for the personal example library.

Run from the repo root:

    python3 -m unittest tests.test_examples
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention import attention_forward
from examples.lib.cues import extract_cues, rule_predict
from examples.lib.dataset import DATASET_DIR, load_split, split_summary
from examples.lib.embeddings import (
    DummyKeyedVectors,
    average_vectors,
    build_padded_sequences,
    concatenate_modalities,
)
from examples.lib.metrics import RECORDED_RESULTS, f1_score, format_results_table, metric_bundle
from examples.lib.tokenize import extract_emojis, tokenize_tweet


class TokenizeTests(unittest.TestCase):
    def test_keeps_hashtag_mention_and_emoji(self) -> None:
        tokens = tokenize_tweet("Hey @user I love Mondays #not 😒")
        self.assertIn("@user", tokens)
        self.assertIn("#not", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("love", tokens)

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize_tweet("   "), [])

    def test_extract_emojis_skips_variation_selectors(self) -> None:
        emojis = extract_emojis("⛳\ufe0f 😒")
        self.assertIn("😒", emojis)
        self.assertTrue(all(ord(char) < 0xFE00 or ord(char) > 0xFE0F for char in emojis))


class CueTests(unittest.TestCase):
    def test_explicit_hashtag_fires(self) -> None:
        decision = rule_predict("Great lecture today #not")
        self.assertEqual(decision.label, 1)
        self.assertEqual(decision.reason, "explicit_hashtag")
        self.assertIn("not", decision.cues.sarcasm_hashtags)

    def test_neutral_tweet_is_negative(self) -> None:
        decision = rule_predict("Had coffee with Sam after lab.")
        self.assertEqual(decision.label, 0)
        self.assertEqual(decision.reason, "no_surface_marker")

    def test_contrast_pattern(self) -> None:
        cues = extract_cues("I love 8am walks 😒")
        self.assertTrue(cues.has_contrast)
        self.assertIn("love", cues.positive_words)


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 3, 5))
        W = rng.normal(size=(5,))
        out = attention_forward(x, W)
        sums = out.weights.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-6)

    def test_mask_zeros_a_step(self) -> None:
        x = np.ones((3, 2))
        W = np.array([1.0, 0.0])
        mask = np.array([1.0, 1.0, 0.0])
        out = attention_forward(x, W, mask=mask)
        self.assertAlmostEqual(float(out.weights[0, -1]), 0.0, places=6)
        self.assertAlmostEqual(float(out.weights[0, :2].sum()), 1.0, places=6)

    def test_two_dimensional_input_is_batched(self) -> None:
        x = np.eye(3)
        W = np.array([1.0, 0.0, 0.0])
        out = attention_forward(x, W)
        self.assertEqual(out.context.shape, (1, 3))
        self.assertEqual(out.weights.shape, (1, 3))

    def test_context_is_weighted_sum(self) -> None:
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        W = np.array([10.0, 0.0])
        out = attention_forward(x, W)
        # First step should dominate because it aligns with W.
        self.assertGreater(out.weights[0, 0], 0.9)
        np.testing.assert_allclose(out.context[0], (out.weights[0, :, None] * x).sum(axis=0))


class EmbeddingTests(unittest.TestCase):
    def test_empty_doc_is_zero_vector(self) -> None:
        model = DummyKeyedVectors.random_from_tokens(["hello"], dim=8, seed=1)
        avg = average_vectors([[]], model)
        self.assertEqual(avg.shape, (1, 8))
        self.assertTrue(np.all(avg == 0))

    def test_concatenation_and_padding(self) -> None:
        docs = [["i", "love", "😒"], ["ok"]]
        model = DummyKeyedVectors.random_from_tokens(["i", "love", "😒", "ok"], dim=4, seed=2)
        word = average_vectors(docs, model)
        emoji = average_vectors(docs, model, predicate=lambda t: t == "😒")
        fused = concatenate_modalities(word, emoji)
        self.assertEqual(fused.shape, (2, 8))
        padded, index = build_padded_sequences(docs, maxlen=4)
        self.assertEqual(padded.shape, (2, 4))
        self.assertEqual(padded[1, 1:].tolist(), [0, 0, 0])
        self.assertEqual(index["i"], 1)
        self.assertTrue(np.all(emoji[1] == 0))


class DatasetTests(unittest.TestCase):
    def test_official_splits_align(self) -> None:
        self.assertTrue(DATASET_DIR.is_dir())
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, n in expected.items():
            split = load_split(name)
            self.assertEqual(len(split), n)
            self.assertEqual(len(split.texts), len(split.labels))
            self.assertTrue(set(split.labels) <= {0, 1})
            summary = split_summary(split)
            self.assertEqual(summary["n"], n)

    def test_test_split_is_balanced(self) -> None:
        split = load_split("test")
        self.assertEqual(sum(split.labels), 1000)
        self.assertAlmostEqual(split.sarcastic_rate(), 0.5)

    def test_unknown_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_split("dev")


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        bundle = metric_bundle([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(bundle["accuracy"], 1.0)
        self.assertEqual(bundle["f1"], 1.0)

    def test_f1_zero_when_no_positive_preds(self) -> None:
        self.assertEqual(f1_score([1, 1, 0], [0, 0, 0]), 0.0)

    def test_recorded_table_lists_every_model(self) -> None:
        table = format_results_table("accuracy")
        for label in ("SVM", "Decision Tree", "Random Forest", "Gradient Boosting", "BiLSTM + Attention"):
            self.assertIn(label, table)
        self.assertEqual(RECORDED_RESULTS["splits"]["train"]["n"], 39780)
        self.assertAlmostEqual(RECORDED_RESULTS["models"]["bilstm_att"]["accuracy"][1], 0.8735)


if __name__ == "__main__":
    unittest.main()
