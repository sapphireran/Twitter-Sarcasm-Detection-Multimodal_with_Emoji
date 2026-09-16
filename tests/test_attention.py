from __future__ import annotations

import math
import unittest

import _path  # noqa: F401
from lib.attention import argmax_token, temporal_attention


class AttentionTests(unittest.TestCase):
    def test_one_hot_weight_selects_matching_step(self):
        sequence = [
            [1.0, 0.0],
            [0.0, 5.0],
            [0.2, 0.1],
        ]
        pooled, alphas, scores = temporal_attention(sequence, weight=[0.0, 1.0])
        self.assertAlmostEqual(sum(alphas), 1.0, places=6)
        self.assertEqual(alphas.index(max(alphas)), 1)
        self.assertGreater(pooled[1], pooled[0])
        self.assertAlmostEqual(scores[1], math.tanh(5.0), places=6)

    def test_mask_zeros_out_padding(self):
        sequence = [
            [0.0, 4.0],
            [8.0, 8.0],
        ]
        _pooled, alphas, _scores = temporal_attention(
            sequence, weight=[1.0, 1.0], mask=[1.0, 0.0]
        )
        self.assertLess(alphas[1], 1e-6)
        self.assertGreater(alphas[0], 0.99)

    def test_bias_can_override_content(self):
        sequence = [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
        _pooled, alphas, _ = temporal_attention(
            sequence, weight=[0.0, 0.0], bias=[0.0, 3.0]
        )
        self.assertGreater(alphas[1], alphas[0])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            temporal_attention([], weight=[1.0])

    def test_argmax_token(self):
        self.assertEqual(argmax_token(["a", "#not", "b"], [0.1, 0.7, 0.2]), "#not")


if __name__ == "__main__":
    unittest.main()
