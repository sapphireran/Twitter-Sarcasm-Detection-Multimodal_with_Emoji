from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.naive_bayes import BernoulliNB


class NaiveBayesTests(unittest.TestCase):
    def test_separates_a_single_feature(self):
        rows = [
            {"leak": 1, "emoji": 0},
            {"leak": 1, "emoji": 1},
            {"leak": 0, "emoji": 0},
            {"leak": 0, "emoji": 0},
        ]
        labels = [1, 1, 0, 0]
        model = BernoulliNB(feature_names=("leak", "emoji"), alpha=1.0).fit(rows, labels)
        self.assertEqual(model.predict_one({"leak": 1, "emoji": 0}), 1)
        self.assertEqual(model.predict_one({"leak": 0, "emoji": 0}), 0)
        top = model.debug_top_features(1)[0][0]
        self.assertEqual(top, "leak")

    def test_rejects_empty_and_bad_alpha(self):
        with self.assertRaises(ValueError):
            BernoulliNB(alpha=0.0)
        with self.assertRaises(ValueError):
            BernoulliNB(feature_names=("a",)).fit([], [])

    def test_unfitted_predict_raises(self):
        model = BernoulliNB(feature_names=("a",))
        with self.assertRaises(RuntimeError):
            model.predict_one({"a": 1})


if __name__ == "__main__":
    unittest.main()
