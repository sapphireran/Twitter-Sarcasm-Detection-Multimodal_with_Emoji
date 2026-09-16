from __future__ import annotations

import unittest

import numpy as np

from examples.lib.attention_numpy import attention_forward
from examples.lib.mean_pool import average_rows, concat_modalities, cosine, lookup_or_skip


class AttentionTests(unittest.TestCase):
    def test_shapes_and_simplex(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 5, 7))
        w = rng.normal(size=(7,))
        b = rng.normal(size=(5,))
        ctx, alpha = attention_forward(x, w, b)
        self.assertEqual(ctx.shape, (4, 7))
        self.assertEqual(alpha.shape, (4, 5))
        np.testing.assert_allclose(alpha.sum(axis=1), np.ones(4), atol=1e-9)

    def test_mask_zeros_pad(self) -> None:
        x = np.ones((1, 3, 2), dtype=np.float64)
        w = np.asarray([1.0, 1.0])
        mask = np.asarray([[1.0, 1.0, 0.0]])
        _, alpha = attention_forward(x, w, mask=mask)
        self.assertAlmostEqual(float(alpha[0, 2]), 0.0, places=12)
        self.assertAlmostEqual(float(alpha[0, :2].sum()), 1.0, places=12)

    def test_known_weights_prefer_second_step(self) -> None:
        x = np.asarray([[[0.9, 0.0], [0.1, 1.0]]], dtype=np.float64)
        w = np.asarray([0.2, 1.4])
        _, alpha = attention_forward(x, w)
        self.assertGreater(alpha[0, 1], alpha[0, 0])

    def test_bad_rank(self) -> None:
        with self.assertRaises(ValueError):
            attention_forward(np.ones((2, 3)), np.ones((3,)))


class MeanPoolTests(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        vec = lookup_or_skip(["zzz"], {"hi": np.ones(3)}, dim=3)
        np.testing.assert_array_equal(vec, np.zeros(3))

    def test_mean_and_concat(self) -> None:
        table = {"a": np.asarray([2.0, 0.0]), "b": np.asarray([0.0, 4.0])}
        word = average_rows([["a", "b"], ["missing"]], table, dim=2)
        emoji = average_rows([["a"], []], table, dim=2)
        fused = concat_modalities(word, emoji)
        np.testing.assert_allclose(word[0], np.asarray([1.0, 2.0]))
        np.testing.assert_array_equal(word[1], np.zeros(2))
        self.assertEqual(fused.shape, (2, 4))
        np.testing.assert_allclose(fused[0], np.asarray([1.0, 2.0, 2.0, 0.0]))

    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])), 1.0)
        self.assertAlmostEqual(cosine(np.zeros(2), np.ones(2)), 0.0)

    def test_concat_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            concat_modalities(np.zeros((2, 3)), np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
