from __future__ import annotations

import unittest

import numpy as np

from examples.lexical import (
    FEATURE_NAMES,
    extract_feature_dict,
    extract_feature_matrix,
    extract_feature_vector,
    standardize,
    top_features_by_abs_weight,
)


class LexicalTests(unittest.TestCase):
    def test_feature_schema_stable(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 16)
        self.assertEqual(len(set(FEATURE_NAMES)), 16)

    def test_vector_order_matches_names(self) -> None:
        sentence = "I LOVEEE monday mornings #not 😒 !!!"
        values = extract_feature_dict(sentence)
        vector = extract_feature_vector(sentence)
        self.assertEqual(list(vector), [values[name] for name in FEATURE_NAMES])
        self.assertGreater(values["n_emoji"], 0)
        self.assertEqual(values["has_not_hashtag"], 1.0)
        self.assertGreaterEqual(values["n_exclaim"], 3)
        self.assertGreaterEqual(values["n_elongated"], 1)
        self.assertGreaterEqual(values["n_allcaps"], 1)

    def test_yeahright_and_sarcasm(self) -> None:
        values = extract_feature_dict("yeah right #SarcasticTweet")
        self.assertEqual(values["has_yeahright"], 1.0)
        self.assertEqual(values["has_sarcasm_hashtag"], 1.0)

    def test_matrix_shape(self) -> None:
        matrix = extract_feature_matrix(["ok", "I love this #not"])
        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))

    def test_empty_matrix(self) -> None:
        matrix = extract_feature_matrix([])
        self.assertEqual(matrix.shape, (0, len(FEATURE_NAMES)))

    def test_standardize_train_is_centered(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(loc=3.0, scale=2.0, size=(40, 5))
        other = rng.normal(size=(3, 5))
        scaled_train, scaled_other = standardize(train, other)
        self.assertTrue(np.allclose(scaled_train.mean(axis=0), 0.0, atol=1e-12))
        self.assertTrue(np.allclose(scaled_train.std(axis=0), 1.0, atol=1e-12))
        self.assertEqual(scaled_other.shape, other.shape)

    def test_constant_column_does_not_nan(self) -> None:
        train = np.ones((10, 2))
        scaled, = standardize(train)
        self.assertFalse(np.isnan(scaled).any())

    def test_top_features(self) -> None:
        weights = np.zeros(len(FEATURE_NAMES))
        weights[FEATURE_NAMES.index("has_not_hashtag")] = -2.5
        weights[FEATURE_NAMES.index("n_emoji")] = 1.1
        ranked = top_features_by_abs_weight(weights, k=2)
        self.assertEqual(ranked[0][0], "has_not_hashtag")
        self.assertEqual(ranked[1][0], "n_emoji")


if __name__ == "__main__":
    unittest.main()
