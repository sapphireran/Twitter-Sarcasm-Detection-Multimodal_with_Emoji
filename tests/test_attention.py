import unittest

import numpy as np

from ccs2lab.attention import raffel_attention, uniform_weights


class AttentionTests(unittest.TestCase):
    def test_uniform_when_rows_identical(self) -> None:
        hidden = np.ones((3, 5, 7))
        weight = np.linspace(-1, 1, 7)
        out = raffel_attention(hidden, weight)
        np.testing.assert_allclose(out.weights, uniform_weights(3, 5), atol=1e-7)
        np.testing.assert_allclose(out.context, hidden[:, 0, :], atol=1e-7)

    def test_mask_zeros_and_renormalizes(self) -> None:
        rng = np.random.default_rng(0)
        hidden = rng.normal(size=(2, 4, 6))
        weight = rng.normal(size=(6,))
        mask = np.ones((2, 4))
        mask[:, 1] = 0
        out = raffel_attention(hidden, weight, mask=mask)
        np.testing.assert_allclose(out.weights[:, 1], 0)
        np.testing.assert_allclose(out.weights.sum(axis=1), np.ones(2), atol=1e-6)

    def test_bias_shape_is_timesteps(self) -> None:
        hidden = np.zeros((1, 4, 3))
        weight = np.array([1.0, 0.0, 0.0])
        hidden[0, 2, 0] = 1.0
        bias = np.array([12.0, 0.0, 0.0, 0.0])
        out = raffel_attention(hidden, weight, bias=bias)
        self.assertGreater(out.weights[0, 0], out.weights[0, 2])

    def test_rejects_bad_weight(self) -> None:
        with self.assertRaises(ValueError):
            raffel_attention(np.ones((1, 2, 3)), np.ones((2,)))

    def test_all_masked_is_finite(self) -> None:
        hidden = np.ones((1, 3, 2))
        weight = np.ones((2,))
        out = raffel_attention(hidden, weight, mask=np.zeros((1, 3)))
        self.assertTrue(np.isfinite(out.context).all())


if __name__ == "__main__":
    unittest.main()
