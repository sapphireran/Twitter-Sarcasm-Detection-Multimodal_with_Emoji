import unittest

import numpy as np

from sarcasm_lib.attention import masked_temporal_attention


class AttentionTests(unittest.TestCase):
    def test_shapes_and_normalization(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 7, 5))
        w = rng.normal(size=(5,))
        out = masked_temporal_attention(x, w)
        self.assertEqual(out.context.shape, (4, 5))
        self.assertEqual(out.weights.shape, (4, 7))
        np.testing.assert_allclose(out.weights.sum(axis=1), np.ones(4), atol=1e-7)

    def test_mask_zeros_padding(self) -> None:
        x = np.ones((1, 4, 2))
        x[0, 3, :] = 99
        w = np.array([1.0, 0.0])
        mask = np.array([[1.0, 1.0, 1.0, 0.0]])
        out = masked_temporal_attention(x, w, mask=mask)
        self.assertAlmostEqual(float(out.weights[0, 3]), 0.0, places=7)
        self.assertAlmostEqual(float(out.weights[0, :3].sum()), 1.0, places=6)

    def test_bias_shifts_mass_to_later_step(self) -> None:
        x = np.ones((1, 3, 1))
        w = np.array([0.0])
        early = masked_temporal_attention(x, w, bias=np.array([2.0, 0.0, 0.0]))
        late = masked_temporal_attention(x, w, bias=np.array([0.0, 0.0, 2.0]))
        self.assertGreater(early.weights[0, 0], early.weights[0, 2])
        self.assertGreater(late.weights[0, 2], late.weights[0, 0])

    def test_rejects_bad_shapes(self) -> None:
        x = np.ones((2, 3, 4))
        with self.assertRaises(ValueError):
            masked_temporal_attention(x, np.ones(3))
        with self.assertRaises(ValueError):
            masked_temporal_attention(np.ones((3, 4)), np.ones(4))

    def test_all_masked_row_stays_finite(self) -> None:
        x = np.ones((1, 3, 2))
        w = np.ones(2)
        mask = np.zeros((1, 3))
        out = masked_temporal_attention(x, w, mask=mask)
        self.assertTrue(np.isfinite(out.context).all())
        self.assertTrue(np.isfinite(out.weights).all())


if __name__ == "__main__":
    unittest.main()
