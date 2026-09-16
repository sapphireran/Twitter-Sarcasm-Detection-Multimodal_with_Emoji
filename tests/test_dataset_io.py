from __future__ import annotations

import unittest

from examples.dataset_io import (
    SPLITS,
    assert_subtest_is_test_emoji,
    cue_stats,
    emoji_subset_indices,
    label_run_count,
    labels_are_blocked,
    load_split,
    summarize_split,
    tweet_has_emoji,
)


class DatasetIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = {name: load_split(name) for name in SPLITS}

    def test_expected_sizes(self) -> None:
        self.assertEqual(self.splits["train"].n, 39780)
        self.assertEqual(self.splits["test"].n, 2000)
        self.assertEqual(self.splits["subtest"].n, 278)

    def test_label_balance(self) -> None:
        train = self.splits["train"]
        test = self.splits["test"]
        subtest = self.splits["subtest"]
        self.assertEqual(train.n_negative, 21292)
        self.assertEqual(train.n_positive, 18488)
        self.assertEqual(test.n_positive, 1000)
        self.assertEqual(test.n_negative, 1000)
        self.assertEqual(subtest.n_positive, 172)
        self.assertEqual(subtest.n_negative, 106)

    def test_eval_labels_are_stored_in_blocks(self) -> None:
        self.assertTrue(labels_are_blocked(self.splits["test"].labels))
        self.assertTrue(labels_are_blocked(self.splits["subtest"].labels))
        self.assertEqual(label_run_count(self.splits["test"].labels), 2)
        self.assertEqual(label_run_count(self.splits["subtest"].labels), 2)

    def test_train_labels_are_mostly_grouped(self) -> None:
        runs = label_run_count(self.splits["train"].labels)
        self.assertEqual(runs, 14)
        self.assertFalse(labels_are_blocked(self.splits["train"].labels))

    def test_unknown_split_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_split("dev")

    def test_subtest_is_emoji_slice_of_test(self) -> None:
        assert_subtest_is_test_emoji(self.splits["test"], self.splits["subtest"])

    def test_every_subtest_row_has_emoji(self) -> None:
        for sentence in self.splits["subtest"].sentences:
            self.assertTrue(tweet_has_emoji(sentence), sentence)

    def test_emoji_subset_count_matches_subtest(self) -> None:
        indices = emoji_subset_indices(self.splits["test"].sentences)
        self.assertEqual(len(indices), self.splits["subtest"].n)

    def test_summary_keys(self) -> None:
        summary = summarize_split(self.splits["test"])
        self.assertEqual(summary["name"], "test")
        self.assertGreater(summary["hashtag_rate"], 0.3)
        self.assertAlmostEqual(summary["emoji_rate"], 278 / 2000)
        self.assertTrue(tweet_has_emoji("I have 3 different types of mad : ⭕ ️"))

    def test_train_emoji_counts_split_by_class(self) -> None:
        stats = cue_stats(self.splits["train"])
        self.assertEqual(stats.has_emoji, stats.emoji_in_negative + stats.emoji_in_positive)
        self.assertGreater(stats.has_emoji, 5000)
        self.assertGreater(stats.has_not_hashtag, 3000)


if __name__ == "__main__":
    unittest.main()
