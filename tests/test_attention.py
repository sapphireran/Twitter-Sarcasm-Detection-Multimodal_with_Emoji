import unittest

import numpy as np

from sarcasm_lib.attention import explain_attention, raffel_attention, softmax


class SoftmaxTests(unittest.TestCase):
    def test_rows_sum_to_one(self) -> None:
        weights = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2), atol=1e-6)

    def test_mask_zeroes_out_padded_steps(self) -> None:
        scores = np.array([[1.0, 1.0, 10.0]])
        mask = np.array([[1.0, 1.0, 0.0]])
        weights = softmax(scores, mask=mask)
        self.assertAlmostEqual(float(weights[0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)


class AttentionTests(unittest.TestCase):
    def test_output_shape_and_focus(self) -> None:
        # One tweet, three tokens, two features. The last token is the cue.
        sequence = np.array(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [4.0, 4.0],
                ]
            ]
        )
        weights = np.array([1.0, 1.0])
        context, alphas, scores = raffel_attention(sequence, weights)
        self.assertEqual(context.shape, (1, 2))
        self.assertEqual(alphas.shape, (1, 3))
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(int(np.argmax(alphas[0])), 2)

    def test_bias_must_match_time_steps(self) -> None:
        sequence = np.zeros((1, 2, 3))
        weights = np.ones(3)
        with self.assertRaises(ValueError):
            raffel_attention(sequence, weights, bias=np.zeros(5))

    def test_walkthrough_top_steps(self) -> None:
        sequence = np.array([[[0.0, 0.0], [2.0, 2.0]]])
        walk = explain_attention(sequence, np.array([1.0, 1.0]))
        top = walk.top_steps(k=1)
        self.assertEqual(top[0][0], 1)


if __name__ == "__main__":
    unittest.main()
