import unittest

import numpy as np

from examples.average_vectors import average_channel, multimodal_features
from examples.fusion import channel_norms, concat_channels, cosine, pairwise_cosine_matrix
from examples.hash_embeddings import HashEmbeddings, default_tables
from examples.tokenize import tweet_tokenize
from examples.toy_corpus import TOY_LABELS, TOY_TWEETS, illustrative_pair


class AverageAndFusionTests(unittest.TestCase):
    def setUp(self):
        self.text, self.emoji = default_tables(dim=32)

    def test_empty_channel_is_zeros(self):
        vec = average_channel(["love", "mondays"], self.emoji)
        self.assertEqual(vec.shape, (32,))
        self.assertTrue(np.allclose(vec, 0.0))

    def test_text_channel_skips_emoji(self):
        tokens = tweet_tokenize("i love mondays 😒")
        vec = average_channel(tokens, self.text)
        only_words = average_channel(["i", "love", "mondays"], self.text)
        self.assertTrue(np.allclose(vec, only_words))

    def test_emoji_channel_uses_only_emoji(self):
        tokens = tweet_tokenize("i love mondays 😒 😭")
        vec = average_channel(tokens, self.emoji)
        two = average_channel(["😒", "😭"], self.emoji)
        self.assertTrue(np.allclose(vec, two))

    def test_hash_embeddings_are_deterministic(self):
        other = HashEmbeddings("text", dim=32)
        self.assertTrue(np.allclose(self.text.embed("love"), other.embed("love")))

    def test_namespaces_do_not_collide(self):
        # Same string hashed in two namespaces must not be identical.
        # (emoji table refuses non-emoji, so compare the raw helper via two tables
        # on an emoji token that both *could* hash — only emoji table accepts it.)
        e1 = HashEmbeddings("emoji", dim=32).embed("😒")
        e2 = HashEmbeddings("text", dim=32)
        # Force-hash the same token in the text namespace by bypassing known().
        from examples.hash_embeddings import _stable_vec

        text_hash = _stable_vec("text", "😒", 32)
        self.assertFalse(np.allclose(e1, text_hash))

    def test_multimodal_width_is_double(self):
        tokenized = [tweet_tokenize(t) for t in TOY_TWEETS]
        x_text, x_multi = multimodal_features(tokenized, self.text, self.emoji)
        self.assertEqual(x_text.shape, (len(TOY_TWEETS), 32))
        self.assertEqual(x_multi.shape, (len(TOY_TWEETS), 64))
        self.assertTrue(np.allclose(x_multi[:, :32], x_text))

    def test_illustrative_pair_emoji_norm(self):
        sarcastic, literal = illustrative_pair()
        tokenized = [tweet_tokenize(sarcastic), tweet_tokenize(literal)]
        _x_text, x_multi = multimodal_features(tokenized, self.text, self.emoji)
        s_text, s_emoji = channel_norms(x_multi[0])
        l_text, l_emoji = channel_norms(x_multi[1])
        self.assertGreater(s_emoji, 0.5)
        self.assertEqual(l_emoji, 0.0)
        self.assertGreater(s_text, 0.0)
        self.assertGreater(l_text, 0.0)
        # Shared words dominate the text half; "#not" keeps cosine below 1.
        self.assertGreater(cosine(x_multi[0, :32], x_multi[1, :32]), 0.85)

    def test_concat_rejects_mismatched_width(self):
        with self.assertRaises(ValueError):
            concat_channels(np.zeros(4), np.zeros(3))

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine(np.zeros(4), np.ones(4)), 0.0)

    def test_pairwise_matrix_shape_and_diagonal(self):
        tokenized = [tweet_tokenize(t) for t in TOY_TWEETS]
        x_text, _ = multimodal_features(tokenized, self.text, self.emoji)
        gram = pairwise_cosine_matrix(x_text)
        self.assertEqual(gram.shape, (len(TOY_TWEETS), len(TOY_TWEETS)))
        self.assertTrue(np.allclose(np.diag(gram), 1.0, atol=1e-6))

    def test_toy_corpus_is_balanced(self):
        self.assertEqual(sum(TOY_LABELS), len(TOY_LABELS) // 2)
        self.assertEqual(len(TOY_TWEETS), 16)
