import unittest

import _paths  # noqa: F401

from sarcasm_lib.metrics import accuracy, binary_f1, confusion, precision, recall


class MetricsTests(unittest.TestCase):
    def test_perfect_scores(self) -> None:
        y = [0, 1, 1, 0]
        self.assertEqual(accuracy(y, y), 1.0)
        self.assertEqual(precision(y, y), 1.0)
        self.assertEqual(recall(y, y), 1.0)
        self.assertEqual(binary_f1(y, y), 1.0)

    def test_confusion_counts(self) -> None:
        table = confusion([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(table.true_neg, 1)
        self.assertEqual(table.false_pos, 1)
        self.assertEqual(table.false_neg, 1)
        self.assertEqual(table.true_pos, 1)

    def test_all_negative_predictions(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        self.assertEqual(precision(y_true, y_pred), 0.0)
        self.assertEqual(recall(y_true, y_pred), 0.0)
        self.assertEqual(binary_f1(y_true, y_pred), 0.0)
        self.assertEqual(accuracy(y_true, y_pred), 0.5)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            confusion([0, 1], [0])
