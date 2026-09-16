from __future__ import annotations

import unittest

import numpy as np

from examples.attention_numpy import (
    demo_sequence,
    demo_weight,
    raffel_attention,
    uniform_pool,
)


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        result = raffel_attention(demo_sequence(), demo_weight())
        self.assertEqual(result.weights.shape, (1, 5))
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)

    def test_sarcasm_step_wins(self) -> None:
        result = raffel_attention(demo_sequence(), demo_weight())
        self.assertEqual(int(result.weights.argmax()), 4)
        self.assertGreater(float(result.weights[0, 4]), 0.35)

    def test_mask_zeros_last_step(self) -> None:
        mask = np.array([1, 1, 1, 1, 0], dtype=np.float64)
        result = raffel_attention(demo_sequence(), demo_weight(), mask=mask)
        self.assertLess(float(result.weights[0, 4]), 1e-6)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)

    def test_bias_length_must_match(self) -> None:
        with self.assertRaises(ValueError):
            raffel_attention(demo_sequence(), demo_weight(), bias=np.zeros(3))

    def test_weight_dim_must_match(self) -> None:
        with self.assertRaises(ValueError):
            raffel_attention(demo_sequence(), np.ones(3))

    def test_batch_and_2d_agree(self) -> None:
        x = demo_sequence()
        w = demo_weight()
        single = raffel_attention(x, w)
        batched = raffel_attention(np.stack([x, x], axis=0), w)
        self.assertEqual(batched.context.shape, (2, 4))
        self.assertTrue(np.allclose(batched.context[0], single.context[0]))

    def test_full_mask_is_finite(self) -> None:
        mask = np.zeros(5)
        result = raffel_attention(demo_sequence(), demo_weight(), mask=mask)
        self.assertTrue(np.isfinite(result.context).all())

    def test_uniform_pool_mask(self) -> None:
        x = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        pooled = uniform_pool(x, mask=np.array([1, 1, 0]))
        self.assertTrue(np.allclose(pooled.reshape(-1), np.array([2.0, 0.0])))

    def test_context_is_convex_combination(self) -> None:
        x = demo_sequence()
        result = raffel_attention(x, demo_weight())
        reconstructed = result.weights @ x
        self.assertTrue(np.allclose(reconstructed, result.context))


if __name__ == "__main__":
    unittest.main()
