import unittest

import numpy as np

from examples.lib.lexical_baseline import binary_metrics, featurize, fit_logistic, sigmoid
from examples.lib.recorded_metrics import markdown_table, rows_for_metric
from examples.lib.tweet_features import FEATURE_NAMES


class LexicalBaselineTests(unittest.TestCase):
    def test_sigmoid_bounds(self) -> None:
        values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertGreater(values[0], 0.0)
        self.assertLess(values[2], 1.0)
        self.assertAlmostEqual(values[1], 0.5, places=6)

    def test_perfectly_separable_not_hashtag(self) -> None:
        texts = [
            "I love this #not",
            "Oh great news #not",
            "dinner with family tonight",
            "rest in peace and love",
        ]
        labels = np.array([1, 1, 0, 0])
        model = fit_logistic(featurize(texts), labels, epochs=400, learning_rate=0.5, l2=1e-5)
        preds = model.predict(featurize(texts))
        np.testing.assert_array_equal(preds, labels)
        weight_map = dict(zip(FEATURE_NAMES, model.weights, strict=True))
        self.assertGreater(weight_map["not_hashtag"], 0.0)

    def test_binary_metrics(self) -> None:
        metrics = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_featurize_shape(self) -> None:
        matrix = featurize(["hello", "love this #not 😒"])
        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))

    def test_recorded_accuracy_table(self) -> None:
        table = markdown_table("accuracy")
        self.assertIn("Bi-LSTM + Attention", table)
        self.assertIn("87.35", table)
        rows = rows_for_metric("accuracy")
        lstm = next(row for row in rows if row.model == "Bi-LSTM + Attention")
        self.assertGreater(lstm.values[1], lstm.values[0])


if __name__ == "__main__":
    unittest.main()
