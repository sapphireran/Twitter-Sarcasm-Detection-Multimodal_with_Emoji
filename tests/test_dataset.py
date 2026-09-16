import unittest
from pathlib import Path

import _paths  # noqa: F401

from sarcasm_lib.dataset import (
    SPLIT_NAMES,
    faithful_readopen_text,
    load_all_splits,
    load_split,
    normalize_sentence,
    read_labels,
    strip_wrapping_quotes,
)

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "dataset"


class DatasetLoaderTests(unittest.TestCase):
    def test_expected_split_sizes(self) -> None:
        splits = load_all_splits()
        self.assertEqual(set(splits), set(SPLIT_NAMES))
        self.assertEqual(len(splits["train"]), 39780)
        self.assertEqual(len(splits["test"]), 2000)
        self.assertEqual(len(splits["test"].labels), 2000)
        self.assertEqual(len(splits["subtest"]), 278)

    def test_label_balance(self) -> None:
        train = load_split("train")
        test = load_split("test")
        sub = load_split("subtest")
        self.assertEqual(train.n_negative, 21292)
        self.assertEqual(train.n_positive, 18488)
        self.assertEqual(test.n_negative, 1000)
        self.assertEqual(test.n_positive, 1000)
        self.assertEqual(sub.n_negative, 106)
        self.assertEqual(sub.n_positive, 172)

    def test_binary_labels_only(self) -> None:
        for split in load_all_splits().values():
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_unknown_split_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_split("dev")

    def test_quote_stripping(self) -> None:
        raw = '"hello, world"'
        self.assertEqual(strip_wrapping_quotes(raw), "hello, world")
        self.assertEqual(normalize_sentence(raw, faithful=False), "hello, world")
        self.assertEqual(faithful_readopen_text(raw), '"hello  world"')

    def test_faithful_mode_changes_commas(self) -> None:
        preserved = load_split("train", faithful=False)
        faithful = load_split("train", faithful=True)
        self.assertEqual(len(preserved), len(faithful))
        # At least one training line contains a comma that ReadOpen would break.
        self.assertTrue(
            any(p != f for p, f in zip(preserved.sentences, faithful.sentences))
        )

    def test_read_labels_skips_blank_lines_consistently_with_files(self) -> None:
        labels = read_labels(DATASET / "test_label.csv")
        self.assertEqual(len(labels), 2000)
        self.assertEqual(labels.count(0), 1000)


if __name__ == "__main__":
    unittest.main()
