from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.embeddings import average_vectors, cosine, embed_tweet, toy_table


class EmbeddingTests(unittest.TestCase):
    def test_unknown_tokens_yield_zeros(self):
        zeros = average_vectors(["zzzz"], {}, dim=4)
        self.assertEqual(zeros, [0.0, 0.0, 0.0, 0.0])

    def test_mean_of_two_known_tokens(self):
        table = {"a": [2.0, 0.0], "b": [0.0, 4.0]}
        self.assertEqual(average_vectors(["a", "b"], table, 2), [1.0, 2.0])

    def test_permutation_invariance(self):
        table = toy_table(8)
        left = embed_tweet("love waiting #not", table, 8)
        right = embed_tweet("#not waiting love", table, 8)
        self.assertAlmostEqual(cosine(left, right), 1.0, places=6)

    def test_concat_width(self):
        table = toy_table(8)
        emoji = {"😑": table["😑"]}
        words = {k: v for k, v in table.items() if k != "😑"}
        vec = embed_tweet("love 😑", words, 8, emoji_table=emoji)
        self.assertEqual(len(vec), 16)

    def test_cosine_orthogonal_and_zero(self):
        self.assertEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
