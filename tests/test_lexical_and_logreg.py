from __future__ import annotations

import unittest

import numpy as np

from examples.lib.lexical_features import (
    FEATURE_NAMES,
    hashtag_rule_predict,
    lexical_feature_matrix,
    lexical_vector,
)
from examples.lib.logreg import LogisticRegressionGD


class LexicalTests(unittest.TestCase):
    def test_vector_length(self) -> None:
        vec = lexical_vector("I love this dirty house #not 😒")
        self.assertEqual(vec.shape, (len(FEATURE_NAMES),))
        self.assertEqual(vec[0], 1.0)  # bias
        names = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(names["has_not_hashtag"], 1.0)
        self.assertGreaterEqual(names["emoji_count"], 1.0)
        self.assertGreaterEqual(names["positive_count"], 1.0)
        self.assertGreaterEqual(names["complaint_count"], 1.0)
        self.assertEqual(names["pos_and_complaint"], 1.0)

    def test_sarcasm_hashtag_flag(self) -> None:
        vec = lexical_vector("great day #SarcasticTweet")
        names = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(names["has_sarcasm_hashtag"], 1.0)
        self.assertEqual(names["has_not_hashtag"], 0.0)

    def test_hashtag_rule(self) -> None:
        self.assertEqual(hashtag_rule_predict("love this #not"), 1)
        self.assertEqual(hashtag_rule_predict("I loovee it #sarcastictweet"), 1)
        self.assertEqual(hashtag_rule_predict("i just imagined you dancing"), 0)

    def test_matrix_stacks(self) -> None:
        mat = lexical_feature_matrix(["a", "b #not"])
        self.assertEqual(mat.shape, (2, len(FEATURE_NAMES)))
        self.assertEqual(mat[1, FEATURE_NAMES.index("has_not_hashtag")], 1.0)

    def test_empty_matrix(self) -> None:
        mat = lexical_feature_matrix([])
        self.assertEqual(mat.shape, (0, len(FEATURE_NAMES)))


class LogregTests(unittest.TestCase):
    def test_separates_and_tag_feature(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        x = np.ones((n, 2))
        x[:, 1] = rng.integers(0, 2, size=n)
        y = x[:, 1].copy()
        model = LogisticRegressionGD(lr=0.5, epochs=200, l2=0.0, seed=0).fit(x, y)
        pred = model.predict(x)
        self.assertGreaterEqual(float((pred == y).mean()), 0.98)
        self.assertGreater(model.weights[1], 0.0)
        self.assertLess(model.loss_history[-1], model.loss_history[0])

    def test_predict_before_fit_raises(self) -> None:
        model = LogisticRegressionGD()
        with self.assertRaises(RuntimeError):
            model.predict(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
