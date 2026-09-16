from __future__ import annotations

import unittest

from _path import EXAMPLES  # noqa: F401
from lib.io import load_all, load_split, repo_root


class IoTests(unittest.TestCase):
    def test_repo_root_finds_dataset(self):
        root = repo_root()
        self.assertTrue((root / "dataset" / "train_sentence.csv").is_file())
        self.assertTrue((root / "data_utils.py").is_file())

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_split("validation")

    def test_alignment_and_counts(self):
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for split in load_all():
            self.assertEqual(len(split), expected[split.name])
            self.assertEqual(len(split.texts), len(split.labels))
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_positive_rates(self):
        train = load_split("train")
        test = load_split("test")
        subtest = load_split("subtest")
        self.assertAlmostEqual(train.positive_rate, 18488 / 39780, places=6)
        self.assertEqual(sum(test.labels), 1000)
        self.assertEqual(sum(subtest.labels), 172)

    def test_quoted_commas_are_rejoined(self):
        train = load_split("train")
        # Line 7 in the raw file is a quoted tweet that contains commas.
        self.assertIn("word requirement", train.texts[6])
        self.assertNotIn('"', train.texts[6][:1])


if __name__ == "__main__":
    unittest.main()
