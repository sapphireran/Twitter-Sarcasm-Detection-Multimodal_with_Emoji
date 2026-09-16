"""Guard the JSON that docs/experiments.md and example 08 both read."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "docs" / "results" / "recorded_metrics.json"

REQUIRED_MODELS = {
    "svm": {"accuracy", "f1", "recall", "precision"},
    "decision_tree": {"accuracy", "f1", "recall", "precision"},
    "random_forest": {"accuracy", "f1", "recall", "precision"},
    "gradient_boosting": {"accuracy", "f1"},
    "bilstm_attention": {"accuracy", "f1", "keras_evaluate"},
}

SLOTS = 4


class RecordedMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    def test_slot_order(self) -> None:
        self.assertEqual(
            self.data["slot_order"],
            ["test_word", "test_word_emoji", "subtest_word", "subtest_word_emoji"],
        )

    def test_split_counts_match_csvs(self) -> None:
        splits = self.data["splits"]
        self.assertEqual(splits["train"]["n"], 39780)
        self.assertEqual(splits["test"]["n"], 2000)
        self.assertEqual(splits["subtest"]["n"], 278)
        self.assertEqual(splits["test"]["positive"], splits["test"]["negative"])

    def test_metric_vector_lengths_and_bounds(self) -> None:
        for name, keys in REQUIRED_MODELS.items():
            block = self.data["models"][name]
            self.assertTrue(keys.issubset(block), msg=name)
            for key in keys:
                if key == "keras_evaluate":
                    continue
                values = block[key]
                self.assertEqual(len(values), SLOTS, msg=f"{name}.{key}")
                for v in values:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    def test_neural_beats_random_forest_on_test_multimodal(self) -> None:
        rf = self.data["models"]["random_forest"]["accuracy"][1]
        nn = self.data["models"]["bilstm_attention"]["accuracy"][1]
        self.assertGreater(nn, rf)

    def test_emoji_helps_neural_on_subtest(self) -> None:
        acc = self.data["models"]["bilstm_attention"]["accuracy"]
        self.assertGreater(acc[3], acc[2])

    def test_keras_evaluate_matches_weekly_rounding(self) -> None:
        keras = self.data["models"]["bilstm_attention"]["keras_evaluate"]
        acc = self.data["models"]["bilstm_attention"]["accuracy"]
        self.assertAlmostEqual(keras["single_modal_test"]["acc"], acc[0], places=3)
        self.assertAlmostEqual(keras["multi_modal_test"]["acc"], acc[1], places=3)
        self.assertAlmostEqual(keras["single_modal_subtest"]["acc"], acc[2], places=3)
        self.assertAlmostEqual(keras["multi_modal_subtest"]["acc"], acc[3], places=3)


if __name__ == "__main__":
    unittest.main()
