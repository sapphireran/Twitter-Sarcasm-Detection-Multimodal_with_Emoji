from __future__ import annotations

import unittest

import numpy as np

from examples.lib.embeddings import (
    KeyedTable,
    average_rows,
    fuse_modalities,
    toy_emoji,
    toy_glove,
)


class EmbeddingTests(unittest.TestCase):
    def test_average_skips_oov_and_zero_fallback(self) -> None:
        table = KeyedTable({"love": np.array([1.0, 0.0])}, dim=2)
        self.assertTrue(np.allclose(average_rows(["love", "xyz"], table), [1.0, 0.0]))
        self.assertTrue(np.allclose(average_rows(["xyz"], table), [0.0, 0.0]))

    def test_average_of_two_rows(self) -> None:
        table = KeyedTable(
            {"a": np.array([2.0, 0.0]), "b": np.array([0.0, 4.0])},
            dim=2,
        )
        self.assertTrue(np.allclose(average_rows(["a", "b"], table), [1.0, 2.0]))

    def test_fuse_concat_dim(self) -> None:
        tokens = ["love", "dirty", "#not", "😒"]
        word, emoji, fused = fuse_modalities(tokens, toy_glove(), toy_emoji())
        self.assertEqual(word.shape, (8,))
        self.assertEqual(emoji.shape, (8,))
        self.assertEqual(fused.shape, (16,))
        self.assertTrue(np.allclose(fused[:8], word))
        self.assertTrue(np.allclose(fused[8:], emoji))
        self.assertGreater(word[2], 0.0)  # #not lives on axis 2
        self.assertGreater(emoji[0], 0.0)  # 😒 lives on axis 0

    def test_vocab_membership_like_gensim(self) -> None:
        table = toy_glove()
        self.assertIn("love", table.vocab)
        self.assertNotIn("zzzz", table.vocab)

    def test_rejects_wrong_dim(self) -> None:
        with self.assertRaises(ValueError):
            KeyedTable({"a": np.array([1.0, 2.0, 3.0])}, dim=2)


if __name__ == "__main__":
    unittest.main()
