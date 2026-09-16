"""Offline tests for the example library and the shipped CSVs.

Run from the repo root::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.attention import AdditiveAttention
from examples.common.bow import CountVectorizer, MultinomialNB, accuracy, f1_binary
from examples.common.emoji import emoji_code_points, extract_emojis
from examples.common.io import SPLIT_NAMES, load_split
from examples.common.tokenize import CUE_HASHTAGS, tokenize_tweet
from examples.common.word2vec import cosine, load_word2vec_binary, lookup_emoji, nearest


class TestDatasetFiles(unittest.TestCase):
    def test_split_lengths_and_labels(self):
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, n in expected.items():
            split = load_split(name)
            self.assertEqual(len(split), n)
            self.assertEqual(set(split.labels), {0, 1})
            self.assertEqual(split.n_literal + split.n_sarcastic, n)

    def test_test_is_balanced(self):
        split = load_split("test")
        self.assertEqual(split.n_literal, 1000)
        self.assertEqual(split.n_sarcastic, 1000)

    def test_subtest_every_row_has_emoji(self):
        split = load_split("subtest")
        missing = [text for text in split.texts if not emoji_code_points(text)]
        self.assertEqual(missing, [])

    def test_flag_regional_indicators(self):
        pts = emoji_code_points("I will not . 🇺 🇸")
        self.assertEqual(pts, ["🇺", "🇸"])

    def test_unknown_split_raises(self):
        with self.assertRaises(ValueError):
            load_split("dev")


class TestTokenize(unittest.TestCase):
    def test_hashtag_and_placeholder_kept(self):
        tokens = tokenize_tweet("I loovee when people text back ... 😒 #sarcastictweet")
        self.assertIn("#sarcastictweet", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("loovee", tokens)

    def test_comma_becomes_space_like_readopen(self):
        tokens = tokenize_tweet("hello,world")
        self.assertEqual(tokens, ["hello", "world"])

    def test_user_placeholder(self):
        tokens = tokenize_tweet("<user> Rest in peace")
        self.assertEqual(tokens[0], "<user>")

    def test_cue_set_covers_common_tags(self):
        self.assertIn("#not", CUE_HASHTAGS)
        self.assertIn("#sarcastictweet", CUE_HASHTAGS)


class TestEmoji(unittest.TestCase):
    def test_extracts_unamused(self):
        self.assertEqual(emoji_code_points("love 😒 #not"), ["😒"])

    def test_repeated_faces(self):
        pts = emoji_code_points("gone. 😢 😢 😢 😢")
        self.assertEqual(pts, ["😢", "😢", "😢", "😢"])

    def test_findall_on_empty(self):
        self.assertEqual(extract_emojis("no pictographs here"), [])


class TestWord2Vec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kv = load_word2vec_binary(ROOT / "emoji2vec_twitter.bin")
        cls.kv300 = load_word2vec_binary(ROOT / "emoji2vec.bin")

    def test_shapes(self):
        self.assertEqual(self.kv.dim, 200)
        self.assertEqual(self.kv300.dim, 300)
        self.assertEqual(len(self.kv), 1661)
        self.assertEqual(len(self.kv300), 1661)

    def test_common_faces_present(self):
        for face in ("😂", "😒", "😍", "😭"):
            self.assertIn(face, self.kv)
            self.assertEqual(self.kv[face].shape, (200,))

    def test_cosine_range(self):
        sim = cosine(self.kv["😒"], self.kv["😒"])
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_nearest_excludes_query(self):
        nbrs = nearest(self.kv["😂"], self.kv, k=5, exclude=["😂"])
        self.assertEqual(len(nbrs), 5)
        self.assertNotIn("😂", [tok for tok, _ in nbrs])

    def test_lookup_heart_variation_selector(self):
        self.assertIsNone(self.kv.index.get("❤"))
        vec = lookup_emoji(self.kv, "❤")
        self.assertIsNotNone(vec)
        np.testing.assert_array_equal(vec, self.kv["❤️"])


class TestAttention(unittest.TestCase):
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(0)
        hidden = rng.normal(size=(2, 4, 6)).astype(np.float32)
        w = rng.normal(size=(6,)).astype(np.float32)
        attn = AdditiveAttention(w=w, bias=np.zeros(4, dtype=np.float32))
        weights = attn.weights(hidden)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2), atol=1e-5)

    def test_mask_zeros_pad(self):
        hidden = np.ones((1, 3, 2), dtype=np.float32)
        w = np.array([1.0, 1.0], dtype=np.float32)
        attn = AdditiveAttention(w=w, bias=None)
        mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        weights = attn.weights(hidden, mask=mask)
        self.assertLess(weights[0, 2], 1e-6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

    def test_peak_recovery(self):
        w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hidden = np.zeros((1, 4, 3), dtype=np.float32)
        hidden[0, 2] = w * 5
        hidden[0, 0] = np.array([0.1, 0, 0])
        attn = AdditiveAttention(w=w, bias=None)
        self.assertEqual(int(attn.weights(hidden)[0].argmax()), 2)


class TestBow(unittest.TestCase):
    def test_separates_obvious_unigrams(self):
        docs = [["love", "this"], ["love", "#not"], ["hate", "this"], ["hate", "#not"]]
        y = np.array([0, 1, 0, 1])
        vec = CountVectorizer.fit(docs, min_count=1, max_features=20)
        x = vec.transform(docs)
        clf = MultinomialNB.fit(x, y)
        pred = clf.predict(x)
        self.assertGreaterEqual(accuracy(y, pred), 0.75)

    def test_f1_all_negative(self):
        y = np.array([0, 0, 0])
        pred = np.array([0, 0, 0])
        self.assertEqual(f1_binary(y, pred), 0.0)


class TestPublishedMetricsCsv(unittest.TestCase):
    def test_csv_has_bilstm_accuracy(self):
        path = ROOT / "docs" / "metrics" / "published_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        acc = [
            r
            for r in rows
            if r["metric"] == "accuracy" and r["model"] == "bilstm_attention"
        ]
        self.assertEqual(len(acc), 1)
        self.assertEqual(float(acc[0]["test_WE"]), 0.8735)
        self.assertAlmostEqual(float(acc[0]["subtest_WE"]), 0.8920863270759583)


if __name__ == "__main__":
    unittest.main()
