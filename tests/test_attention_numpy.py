import unittest

import numpy as np

from examples.lib.attention_numpy import (
    attention_pool,
    attention_scores,
    masked_softmax,
    uniform_attention_pool,
)


class AttentionNumpyTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self) -> None:
        scores = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        weights = masked_softmax(scores)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2), atol=1e-6)

    def test_mask_zeros_out_padded_steps(self) -> None:
        scores = np.array([[10.0, 10.0, 10.0]])
        mask = np.array([[1.0, 1.0, 0.0]])
        weights = masked_softmax(scores, mask=mask)
        np.testing.assert_allclose(weights[0, 2], 0.0, atol=1e-8)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(1), atol=1e-6)

    def test_attention_is_weighted_sum(self) -> None:
        sequences = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        weight = np.array([1.0, 0.0])
        context, attn = attention_pool(sequences, weight)
        expected = (attn[0, :, None] * sequences[0]).sum(axis=0)
        np.testing.assert_allclose(context[0], expected, atol=1e-6)
        self.assertEqual(context.shape, (1, 2))

    def test_bias_changes_scores(self) -> None:
        sequences = np.ones((1, 3, 2))
        weight = np.array([0.2, 0.2])
        plain = attention_scores(sequences, weight)
        biased = attention_scores(sequences, weight, bias=np.array([0.0, 0.0, 2.0]))
        self.assertGreater(biased[0, 2], plain[0, 2])

    def test_flip_token_can_dominate(self) -> None:
        sequences = np.array(
            [[[0.9, 0.0, 0.0], [0.1, 0.8, 0.0], [0.0, 0.1, 1.0]]],
            dtype=np.float64,
        )
        weight = np.array([0.1, 0.2, 1.5])
        _context, attn = attention_pool(sequences, weight)
        self.assertEqual(int(np.argmax(attn[0])), 2)

    def test_uniform_pool_is_mean(self) -> None:
        sequences = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        pooled = uniform_attention_pool(sequences)
        np.testing.assert_allclose(pooled, np.array([[2.0, 3.0]]))

    def test_rejects_wrong_shapes(self) -> None:
        with self.assertRaises(ValueError):
            attention_pool(np.ones((2, 3)), np.ones(3))
        with self.assertRaises(ValueError):
            attention_pool(np.ones((1, 2, 3)), np.ones(4))


if __name__ == "__main__":
    unittest.main()
