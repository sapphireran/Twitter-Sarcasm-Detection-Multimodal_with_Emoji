from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.lib.dataset_io import (
    load_all_splits,
    load_split,
    overlap_count,
    read_labels,
    read_sentences,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_T = ROOT / "examples" / "fixtures" / "sample_tweets.csv"
FIXTURE_L = ROOT / "examples" / "fixtures" / "sample_labels.csv"


class DatasetIoTests(unittest.TestCase):
    def test_fixture_roundtrip(self) -> None:
        split = load_split(FIXTURE_T, FIXTURE_L, name="fixture")
        self.assertEqual(len(split), 10)
        self.assertEqual(sum(split.labels), 5)
        self.assertAlmostEqual(split.positive_rate(), 0.5)
        self.assertEqual(len(list(split.pairs())), 10)

    def test_comma_flatten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sent = Path(tmp) / "s.csv"
            lab = Path(tmp) / "l.csv"
            sent.write_text('"hello, world"\nplain\n', encoding="utf-8")
            lab.write_text("1\n0\n", encoding="utf-8")
            split = load_split(sent, lab)
        self.assertEqual(split.texts[0], '"hello  world"')
        self.assertEqual(split.labels, [1, 0])

    def test_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sent = Path(tmp) / "s.csv"
            lab = Path(tmp) / "l.csv"
            sent.write_text("a\nb\n", encoding="utf-8")
            lab.write_text("1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split(sent, lab)

    def test_overlap(self) -> None:
        split = load_split(FIXTURE_T, FIXTURE_L, name="a")
        self.assertEqual(overlap_count(split, split), 10)

    def test_real_dataset_alignment(self) -> None:
        splits = load_all_splits(ROOT / "dataset")
        self.assertEqual(len(splits["train"]), 39780)
        self.assertEqual(len(splits["test"]), 2000)
        self.assertEqual(len(splits["subtest"]), 278)
        self.assertEqual(overlap_count(splits["subtest"], splits["test"]), 278)
        self.assertEqual(sum(splits["test"].labels), 1000)
        self.assertEqual(read_labels(ROOT / "dataset" / "train_label.csv")[0], 0)
        self.assertTrue(read_sentences(ROOT / "dataset" / "train_sentence.csv")[0])


if __name__ == "__main__":
    unittest.main()
