import unittest

import numpy as np

from examples.attention_numpy import (
    AttentionWeights,
    attention_forward,
    peaked_weights,
    softmax,
)


class AttentionTests(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        probs = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_peaked_attention_picks_spike(self):
        hidden = np.zeros((8, 4))
        hidden[5] = 1.0
        context, alphas = attention_forward(hidden, peaked_weights(8, 4, peak=5))
        self.assertEqual(int(alphas.argmax()), 5)
        self.assertGreater(alphas[5], 0.5)
        self.assertAlmostEqual(float(alphas.sum()), 1.0, places=6)
        # Context should look like the spiked step.
        self.assertTrue(np.all(context > 0.4))

    def test_mask_zeros_pad_step(self):
        hidden = np.ones((5, 3))
        hidden[0] = 3.0
        weights = AttentionWeights(W=np.ones(3), b=None)
        mask = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        _ctx, alphas = attention_forward(hidden, weights, mask=mask)
        self.assertLess(alphas[0], 1e-6)
        self.assertAlmostEqual(float(alphas.sum()), 1.0, places=5)

    def test_batch_and_unbatched_match(self):
        rng = np.random.default_rng(0)
        hidden = rng.normal(size=(3, 7, 5))
        weights = AttentionWeights(W=np.linspace(-0.4, 0.4, 5), b=np.zeros(7))
        ctx_b, a_b = attention_forward(hidden, weights)
        ctx_0, a_0 = attention_forward(hidden[0], weights)
        self.assertEqual(ctx_b.shape, (3, 5))
        self.assertTrue(np.allclose(ctx_b[0], ctx_0))
        self.assertTrue(np.allclose(a_b[0], a_0))

    def test_flat_hidden_is_uniform(self):
        hidden = np.ones((4, 2))
        _ctx, alphas = attention_forward(hidden, AttentionWeights(W=np.ones(2)))
        self.assertTrue(np.allclose(alphas, 0.25, atol=1e-6))

    def test_rejects_bad_weight_shapes(self):
        hidden = np.ones((3, 2))
        with self.assertRaises(ValueError):
            attention_forward(hidden, AttentionWeights(W=np.ones(3)))

    def test_matches_manual_formula(self):
        hidden = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        W = np.array([0.5, -0.25])
        b = np.array([0.1, 0.0, -0.1])
        scores = np.tanh(hidden @ W + b)
        scores = scores - scores.max()
        exp = np.exp(scores)
        expected = exp / (exp.sum() + np.finfo(exp.dtype).eps)
        _ctx, alphas = attention_forward(hidden, AttentionWeights(W=W, b=b))
        self.assertTrue(np.allclose(alphas, expected))
