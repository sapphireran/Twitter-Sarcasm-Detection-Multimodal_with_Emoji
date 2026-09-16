"""Readers for the project's ``dataset/*.csv`` files.

Two readers are provided on purpose:

* ``read_open_replica`` follows ``data_utils.ReadOpen`` line-for-line so
  examples can show what the 2023 training code actually saw.
* ``read_sentence_strings`` uses the ``csv`` module so analysis scripts
  keep commas that live inside quoted tweets.
"""

from __future__ import annotations

import csv
from pathlib import Path

from examples.tokenize import join_comma_split_line, tokenize_tweet

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

SPLIT_FILES = {
    "train": (DATASET_DIR / "train_sentence.csv", DATASET_DIR / "train_label.csv"),
    "test": (DATASET_DIR / "test_sentence.csv", DATASET_DIR / "test_label.csv"),
    "subtest": (DATASET_DIR / "subtest_sentence.csv", DATASET_DIR / "subtest_label.csv"),
}


def read_labels(path: str | Path) -> list[int]:
    """Read a one-column 0/1 label file (no header)."""
    labels: list[int] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            labels.append(int(text.split(",")[0]))
    return labels


def read_open_replica(
    sentence_path: str | Path,
    label_path: str | Path,
    lowercase: bool = True,
) -> tuple[list[list[str]], list[int], int]:
    """Replica of ``data_utils.ReadOpen`` that does not import NLTK or pandas.

    Returns ``(tokenized_docs, labels, line_count)``.
    """
    with Path(sentence_path).open(encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    docs: list[list[str]] = []
    for line in lines:
        sentence = join_comma_split_line(line)
        docs.append(tokenize_tweet(sentence, lowercase=lowercase))

    labels = read_labels(label_path)
    return docs, labels, len(lines)


def read_sentence_strings(path: str | Path) -> list[str]:
    """Read one tweet per line, honoring CSV quotes.

    A line like ``"So many useless classes , great to be student"`` is kept
    as a single string with its comma. This is the reader analysis scripts
    should use; it is *not* what ``ReadOpen`` did in 2023.
    """
    texts: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                texts.append("")
                continue
            texts.append(",".join(row) if len(row) > 1 else row[0])
    return texts


def iter_split(name: str) -> tuple[list[str], list[int]]:
    """Return ``(raw_texts, labels)`` for ``train``, ``test``, or ``subtest``."""
    if name not in SPLIT_FILES:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(SPLIT_FILES)}")
    sentence_path, label_path = SPLIT_FILES[name]
    return read_sentence_strings(sentence_path), read_labels(label_path)
