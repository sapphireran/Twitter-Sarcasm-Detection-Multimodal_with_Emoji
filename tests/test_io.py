import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sarcasm_lib.io import load_split, read_labels, read_sentences, read_sentences_legacy


class SentenceReaderTests(unittest.TestCase):
    def test_csv_reader_keeps_commas_inside_quotes(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tweets.csv"
            path.write_text(
                '"Late nights, early mornings"\nplain tweet without commas\n',
                encoding="utf-8",
            )
            texts = read_sentences(path)
        self.assertEqual(
            texts,
            ["Late nights, early mornings", "plain tweet without commas"],
        )

    def test_legacy_reader_turns_commas_into_spaces(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tweets.csv"
            path.write_text(
                '"Late nights, early mornings"\n',
                encoding="utf-8",
            )
            texts = read_sentences_legacy(path)
        self.assertEqual(texts, ['"Late nights  early mornings"'])

    def test_read_labels_skips_blank_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("0\n\n1\n", encoding="utf-8")
            self.assertEqual(read_labels(path), [0, 1])


class RealSplitTests(unittest.TestCase):
    def test_official_splits_are_aligned(self) -> None:
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, n in expected.items():
            split = load_split(name)
            self.assertEqual(len(split), n)
            self.assertEqual(len(split.texts), len(split.labels))
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_test_split_is_balanced(self) -> None:
        split = load_split("test")
        self.assertEqual(split.positive_count, 1000)
        self.assertEqual(split.negative_count, 1000)

    def test_unknown_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_split("dev")


if __name__ == "__main__":
    unittest.main()
