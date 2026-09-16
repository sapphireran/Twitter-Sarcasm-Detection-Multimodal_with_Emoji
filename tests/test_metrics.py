from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.metrics import accuracy, confusion, f1, majority_baseline, precision, recall


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        y = [0, 1, 1, 0]
        self.assertEqual(accuracy(y, y), 1.0)
        self.assertEqual(f1(y, y), 1.0)
        self.assertEqual(confusion(y, y), (2, 0, 2, 0))

    def test_all_negative_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        self.assertEqual(precision(y_true, y_pred), 0.0)
        self.assertEqual(recall(y_true, y_pred), 0.0)
        self.assertEqual(f1(y_true, y_pred), 0.0)
        self.assertEqual(accuracy(y_true, y_pred), 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            accuracy([0, 1], [0])

    def test_majority_baseline(self):
        self.assertEqual(majority_baseline([0, 0, 1]), 2 / 3)
        self.assertEqual(majority_baseline([]), 0.0)


if __name__ == "__main__":
    unittest.main()
