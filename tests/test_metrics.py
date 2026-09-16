import unittest

from ccs2lab.metrics import binary_scores, odds_ratio, wilson_interval


class MetricTests(unittest.TestCase):
    def test_perfect(self) -> None:
        scores = binary_scores([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(scores.accuracy, 1.0)
        self.assertEqual(scores.f1, 1.0)

    def test_all_negative_predictions(self) -> None:
        scores = binary_scores([0, 1, 1], [0, 0, 0])
        self.assertEqual(scores.precision, 0.0)
        self.assertEqual(scores.recall, 0.0)
        self.assertEqual(scores.tp, 0)
        self.assertEqual(scores.fn, 2)

    def test_wilson_bounds(self) -> None:
        p, lo, hi = wilson_interval(515, 2000)
        self.assertAlmostEqual(p, 0.2575)
        self.assertLess(lo, p)
        self.assertGreater(hi, p)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_rejects_bad_k(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)

    def test_odds_ratio_infinite_avoided_by_smoothing(self) -> None:
        ratio = odds_ratio(10, 0, 1, 10)
        self.assertGreater(ratio, 1.0)
        self.assertTrue(ratio < 1e6)


if __name__ == "__main__":
    unittest.main()
