from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.lib.dataset_io import (
    cue_stats,
    load_split,
    read_labels,
    read_sentences,
    split_sizes,
    top_hashtags,
)

ROOT = Path(__file__).resolve().parents[1]


class DatasetIoTests(unittest.TestCase):
    def test_real_splits_are_aligned(self) -> None:
        expected = {
            "train": (39780, 21292, 18488),
            "test": (2000, 1000, 1000),
            "subtest": (278, 106, 172),
        }
        for name, (rows, literal, sarcastic) in expected.items():
            split = load_split(name, ROOT / "dataset")
            sizes = split_sizes(split)
            self.assertEqual(sizes["rows"], rows, name)
            self.assertEqual(sizes["literal"], literal, name)
            self.assertEqual(sizes["sarcastic"], sarcastic, name)

    def test_subtest_is_emoji_heavy(self) -> None:
        split = load_split("subtest", ROOT / "dataset")
        cues = cue_stats(split)
        self.assertGreaterEqual(cues["with_emoji"], 270)
        self.assertEqual(cues["with_emoji"] + (len(split) - cues["with_emoji"]), len(split))

    def test_test_top_hashtag_is_not(self) -> None:
        split = load_split("test", ROOT / "dataset")
        top = top_hashtags(split, n=1)
        self.assertEqual(top[0][0], "#not")
        self.assertGreaterEqual(top[0][1], 400)

    def test_mismatched_files_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo_sentence.csv").write_text("hello\nworld\n", encoding="utf-8")
            (root / "demo_label.csv").write_text("0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split("demo", root)

    def test_bad_label_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("0\n2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_labels(path)

    def test_read_sentences_skips_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("one\n\n two \n", encoding="utf-8")
            self.assertEqual(read_sentences(path), ["one", " two "])


if __name__ == "__main__":
    unittest.main()
