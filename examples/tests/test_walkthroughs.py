"""Unit tests for the personal walkthrough helpers."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.attention_demo import attention
from examples.heuristic_baseline import metrics, predict
from examples.lib.dataset import load_all, load_split
from examples.lib.tokenize import comma_unwrap, pad_sequences, tokenize_tweet
from examples.report_metrics import METRICS, we_minus_w


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = load_all()

    def test_expected_sizes(self) -> None:
        self.assertEqual(self.splits["train"].n, 39780)
        self.assertEqual(self.splits["test"].n, 2000)
        self.assertEqual(self.splits["subtest"].n, 278)

    def test_labels_are_binary(self) -> None:
        for split in self.splits.values():
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_test_is_balanced(self) -> None:
        labels = self.splits["test"].labels
        self.assertEqual(labels.count(0), 1000)
        self.assertEqual(labels.count(1), 1000)

    def test_subtest_all_non_ascii(self) -> None:
        for sentence in self.splits["subtest"].sentences:
            self.assertTrue(any(ord(ch) > 127 for ch in sentence), sentence[:80])


class TokenizeTests(unittest.TestCase):
    def test_comma_unwrap_matches_readopen(self) -> None:
        raw = '"So many useless classes , great to be student"'
        self.assertEqual(
            comma_unwrap(raw),
            '"So many useless classes   great to be student"',
        )

    def test_hashtag_and_user_and_emoji_survive(self) -> None:
        tokens = tokenize_tweet("<user> I love this #not 😒")
        self.assertIn("<user>", tokens)
        self.assertIn("#not", tokens)
        self.assertTrue(any(ord(tok[0]) > 127 for tok in tokens if tok))

    def test_lowercase(self) -> None:
        tokens = tokenize_tweet("Yay #NOT")
        self.assertIn("#not", tokens)
        self.assertNotIn("#NOT", tokens)

    def test_pad_sequences_post(self) -> None:
        padded = pad_sequences([[1, 2, 3], [4]], maxlen=3)
        self.assertEqual(padded, [[1, 2, 3], [4, 0, 0]])


class HeuristicTests(unittest.TestCase):
    def test_tag_predicts_positive(self) -> None:
        self.assertEqual(predict("wow this is fine #sarcasm"), 1)
        self.assertEqual(predict("wow this is fine"), 0)

    def test_perfect_predictions_score_one(self) -> None:
        scores = metrics([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["f1"], 1.0)

    def test_test_precision_is_one_for_explicit_tags(self) -> None:
        # On the checked-in test file, every #not/#sarcasm tweet is label 1.
        split = load_split("test")
        preds = [predict(s) for s in split.sentences]
        scores = metrics(split.labels, preds)
        self.assertEqual(scores["fp"], 0.0)
        self.assertGreater(scores["tp"], 500)


class AttentionTests(unittest.TestCase):
    def test_row_sums_to_one(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 5, 8))
        w = rng.normal(size=(8,))
        _ctx, alpha, _e = attention(x, w)
        np.testing.assert_allclose(alpha.sum(axis=1), np.ones(3), atol=1e-6)

    def test_mask_zeros_a_step(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1, 4, 3))
        w = rng.normal(size=(3,))
        mask = np.array([[1.0, 1.0, 0.0, 1.0]])
        _ctx, alpha, _e = attention(x, w, mask=mask)
        self.assertLess(alpha[0, 2], 1e-12)

    def test_all_masked_stays_finite(self) -> None:
        x = np.ones((1, 3, 2))
        w = np.ones(2)
        mask = np.zeros((1, 3))
        ctx, alpha, _e = attention(x, w, mask=mask)
        self.assertTrue(np.isfinite(ctx).all())
        self.assertTrue(np.isfinite(alpha).all())


class MetricsTableTests(unittest.TestCase):
    def test_deep_model_is_best_on_test_we(self) -> None:
        deep = METRICS["bilstm_attention"]["accuracy"][1]
        others = [METRICS[k]["accuracy"][1] for k in METRICS if k != "bilstm_attention"]
        self.assertGreater(deep, max(others))

    def test_rf_subtest_gain_larger_than_test_gain(self) -> None:
        deltas = we_minus_w(METRICS["random_forest"]["accuracy"])
        self.assertGreater(deltas["delta_sub"], deltas["delta_test"])
        self.assertGreater(deltas["delta_sub"], 0.04)

    def test_svm_test_delta_is_negative(self) -> None:
        deltas = we_minus_w(METRICS["svm"]["accuracy"])
        self.assertLess(deltas["delta_test"], 0)


if __name__ == "__main__":
    unittest.main()
