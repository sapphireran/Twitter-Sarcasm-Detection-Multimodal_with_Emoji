import unittest

from examples.dataset_stats import SPLIT_FILES, summarize_pairs, summarize_split
from examples.tokenize import read_pairs


class DatasetStatsTests(unittest.TestCase):
    def test_split_files_exist(self):
        for name, (sent, lab) in SPLIT_FILES.items():
            self.assertTrue(sent.is_file(), name)
            self.assertTrue(lab.is_file(), name)

    def test_train_counts(self):
        summary = summarize_split("train")
        self.assertEqual(summary.n, 39780)
        self.assertEqual(summary.n_sarcastic, 18488)
        self.assertEqual(summary.n_literal, 21292)
        self.assertAlmostEqual(summary.sarcasm_rate, 18488 / 39780, places=5)

    def test_test_is_balanced(self):
        summary = summarize_split("test")
        self.assertEqual(summary.n, 2000)
        self.assertEqual(summary.n_sarcastic, 1000)

    def test_subtest_is_sarcasm_heavy_and_marker_heavy(self):
        train = summarize_split("train")
        sub = summarize_split("subtest")
        self.assertEqual(sub.n, 278)
        self.assertGreater(sub.sarcasm_rate, train.sarcasm_rate)
        self.assertGreater(sub.emoji_rate, train.emoji_rate)
        self.assertGreater(sub.sarcasm_marker_rate, train.sarcasm_marker_rate)
        # Subtest is an emoji slice: almost every row has a pictograph.
        self.assertGreater(sub.emoji_rate, 0.95)
        self.assertGreater(sub.sarcasm_marker_rate, 0.4)

    def test_pairs_align(self):
        sent, lab = SPLIT_FILES["subtest"]
        pairs = read_pairs(sent, lab)
        self.assertEqual(len(pairs), 278)
        self.assertEqual(pairs[0][1], 1)

    def test_summarize_pairs_on_tiny_set(self):
        rows = [
            ("i love this 😒 #not", 1),
            ("have a nice day", 0),
        ]
        summary = summarize_pairs("tiny", rows, top_k=3)
        self.assertEqual(summary.n, 2)
        self.assertEqual(summary.n_sarcastic, 1)
        self.assertEqual(summary.emoji_rate, 0.5)
        self.assertEqual(summary.hashtag_rate, 0.5)
        self.assertEqual(summary.top_hashtags[0][0], "#not")
