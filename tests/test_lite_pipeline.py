"""Unit tests for examples/lite_pipeline.py (NumPy + stdlib only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (
    CUE_HASHTAGS,
    accuracy,
    build_embed_table,
    cue_emoji_features,
    docs_to_padded,
    has_cue_hashtag,
    is_emoji_token,
    mean_pool,
    mimic_readopen_commas,
    nearest_centroid_predict,
    pooled_views,
    raffel_attention,
    read_sentence_label_pair,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    def test_hashtag_and_emoji_stay_whole(self) -> None:
        tokens = tokenize("I loovee it 😒 #sarcastictweet")
        self.assertIn("#sarcastictweet", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("loovee", tokens)

    def test_user_placeholder(self) -> None:
        tokens = tokenize("<user> Rest in peace")
        self.assertEqual(tokens[0], "<user>")

    def test_lowercases(self) -> None:
        self.assertIn("#not", tokenize("Fine #Not"))

    def test_comma_smash_drops_comma_token(self) -> None:
        raw = "So many classes , great to be student"
        keep = tokenize(raw)
        smash = tokenize(mimic_readopen_commas(raw))
        self.assertIn(",", keep)
        self.assertNotIn(",", smash)


class EmojiAndCueTests(unittest.TestCase):
    def test_emoji_predicate(self) -> None:
        self.assertTrue(is_emoji_token("😭"))
        self.assertTrue(is_emoji_token("😒"))
        self.assertFalse(is_emoji_token("#not"))
        self.assertFalse(is_emoji_token("love"))

    def test_cue_set_covers_docs(self) -> None:
        for tag in ("#not", "#sarcasm", "#sarcastictweet", "#yeahright"):
            self.assertIn(tag, CUE_HASHTAGS)
            self.assertTrue(has_cue_hashtag([tag, "wow"]))


class PoolingTests(unittest.TestCase):
    def test_mean_pool_average_and_zero_fallback(self) -> None:
        table = {"good": np.array([2.0, 0.0]), "bad": np.array([0.0, 2.0])}
        avg = mean_pool(["good", "bad", "missing"], table, width=2)
        np.testing.assert_allclose(avg, [1.0, 1.0])
        z = mean_pool(["missing"], table, width=2)
        np.testing.assert_allclose(z, [0.0, 0.0])

    def test_pooled_views_concat(self) -> None:
        docs = [["hello", "😭"], ["hello"]]
        words = {"hello": np.ones(3)}
        emoji = {"😭": np.full(3, 2.0)}
        x_w, x_m = pooled_views(docs, words, emoji, word_width=3, emoji_width=3)
        self.assertEqual(x_w.shape, (2, 3))
        self.assertEqual(x_m.shape, (2, 6))
        np.testing.assert_allclose(x_m[0, 3:], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(x_m[1, 3:], [0.0, 0.0, 0.0])

    def test_embed_table_uses_vocab_size_not_tweet_count(self) -> None:
        docs = [["a", "b"], ["a"]]
        words = {"a": np.ones(4), "b": np.full(4, -1.0)}
        vocab, matrix = build_embed_table(docs, words, {}, width=4)
        self.assertEqual(len(vocab), 2)
        self.assertEqual(matrix.shape, (3, 4))
        np.testing.assert_allclose(matrix[0], 0)
        padded = docs_to_padded(docs, vocab, maxlen=3)
        self.assertEqual(padded.shape, (2, 3))
        self.assertEqual(padded[0, 2], 0)


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one_and_reconstruct(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 5, 6))
        w = rng.normal(size=(6,))
        ctx, attn = raffel_attention(x, w)
        np.testing.assert_allclose(attn.sum(axis=1), np.ones(3), atol=1e-6)
        recon = (x * attn[..., None]).sum(axis=1)
        np.testing.assert_allclose(recon, ctx)

    def test_mask_zeros_step(self) -> None:
        x = np.ones((1, 4, 2))
        x[0, 2] = 5.0
        w = np.array([1.0, 1.0])
        mask = np.array([[1.0, 1.0, 0.0, 1.0]])
        _, attn = raffel_attention(x, w, mask=mask)
        self.assertLess(attn[0, 2], 1e-8)
        self.assertAlmostEqual(float(attn[0].sum()), 1.0, places=6)

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            raffel_attention(np.zeros((2, 3)), np.zeros((3,)))


class BaselineTests(unittest.TestCase):
    def test_nearest_centroid_separates_two_blobs(self) -> None:
        x_tr = np.array([[0.0, 0.0], [0.1, -0.1], [5.0, 5.0], [5.1, 4.9]])
        y_tr = np.array([0, 0, 1, 1])
        x_te = np.array([[0.2, 0.0], [4.8, 5.2]])
        pred = nearest_centroid_predict(x_tr, y_tr, x_te)
        np.testing.assert_array_equal(pred, [0, 1])
        self.assertEqual(accuracy(np.array([0, 1]), pred), 1.0)

    def test_cue_features_width(self) -> None:
        feats = cue_emoji_features([["#not", "lol"], ["plain", "text"]])
        self.assertEqual(feats.shape, (2, 4))
        self.assertEqual(feats[0, 0], 1.0)
        self.assertEqual(feats[1, 0], 0.0)


class FixtureAlignmentTests(unittest.TestCase):
    def test_tiny_fixture(self) -> None:
        base = ROOT / "examples" / "fixtures"
        docs, labels = read_sentence_label_pair(
            base / "tiny_sentence.csv", base / "tiny_label.csv"
        )
        self.assertEqual(len(docs), 12)
        self.assertEqual(len(labels), 12)
        self.assertEqual(int(labels.sum()), 6)
        self.assertTrue(any(is_emoji_token(t) for t in docs[0]))


if __name__ == "__main__":
    unittest.main()
