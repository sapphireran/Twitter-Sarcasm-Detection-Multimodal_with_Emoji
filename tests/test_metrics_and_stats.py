import unittest

from sarcasm_lib.io import load_split
from sarcasm_lib.metrics import binary_metrics, confusion_counts
from sarcasm_lib.stats import compute_split_stats


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = binary_metrics([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.f1, 1.0)

    def test_all_negative_predictions_have_zero_recall(self) -> None:
        metrics = binary_metrics([1, 1, 0], [0, 0, 0])
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.counts.false_negative, 2)

    def test_rejects_non_binary_labels(self) -> None:
        with self.assertRaises(ValueError):
            confusion_counts([0, 2], [0, 1])


class StatsTests(unittest.TestCase):
    def test_train_counts_match_files(self) -> None:
        stats = compute_split_stats(load_split("train"), top_k=5)
        self.assertEqual(stats.n, 39780)
        self.assertEqual(stats.n_sarcastic, 18488)
        self.assertEqual(stats.n_non_sarcastic, 21292)
        self.assertGreater(stats.n_with_marker_hashtag, 3000)
        self.assertGreater(stats.marker_precision, 0.9)
        self.assertTrue(stats.top_hashtags)
        self.assertEqual(stats.top_hashtags[0][0], "#not")


if __name__ == "__main__":
    unittest.main()
