from __future__ import annotations

import unittest

import numpy as np

from examples.lib.attention_numpy import demo_sequence, raffel_attention


class AttentionTests(unittest.TestCase):
    def test_shapes_and_softmax(self) -> None:
        demo = demo_sequence()
        context, alpha = raffel_attention(
            demo["x"], demo["weight"], bias=demo["bias"], mask=demo["mask"]
        )
        self.assertEqual(context.shape, (1, 4))
        self.assertEqual(alpha.shape, (1, 5))
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=6)

    def test_mass_on_late_cues(self) -> None:
        demo = demo_sequence()
        _, alpha = raffel_attention(
            demo["x"], demo["weight"], bias=demo["bias"], mask=demo["mask"]
        )
        late = float(alpha[0, 3] + alpha[0, 4])
        early = float(alpha[0, 0] + alpha[0, 1] + alpha[0, 2])
        self.assertGreater(late, early)
        self.assertGreater(late, 0.55)

    def test_mask_renormalizes(self) -> None:
        demo = demo_sequence()
        mask = demo["mask"].copy()
        mask[0, -1] = 0.0
        _, alpha = raffel_attention(
            demo["x"], demo["weight"], bias=demo["bias"], mask=mask
        )
        self.assertAlmostEqual(float(alpha[0, -1]), 0.0, places=6)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=6)

    def test_empty_mask_does_not_nan(self) -> None:
        x = np.ones((1, 3, 2))
        weight = np.array([1.0, 0.0])
        mask = np.zeros((1, 3))
        context, alpha = raffel_attention(x, weight, mask=mask)
        self.assertFalse(np.isnan(context).any())
        self.assertFalse(np.isnan(alpha).any())

    def test_rejects_bad_weight(self) -> None:
        with self.assertRaises(ValueError):
            raffel_attention(np.ones((1, 2, 3)), np.ones((2,)))


if __name__ == "__main__":
    unittest.main()
