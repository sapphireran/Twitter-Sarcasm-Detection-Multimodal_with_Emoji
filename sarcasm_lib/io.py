"""Dataset readers that do not depend on pandas or NLTK.

The 2023 notebooks load tweets with ``data_utils.ReadOpen``, which splits each
line on commas and rejoins the pieces. That works for this corpus but mangles
commas that belong to the tweet text. The helpers here keep a CSV-aware reader
as the default and expose the legacy behaviour for side-by-side examples.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .paths import SPLIT_FILES


@dataclass(frozen=True)
class DatasetSplit:
    """Aligned tweet texts and binary sarcasm labels."""

    name: str
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def positive_count(self) -> int:
        return sum(self.labels)

    @property
    def negative_count(self) -> int:
        return len(self.labels) - self.positive_count

    def pairs(self) -> Iterable[tuple[str, int]]:
        return zip(self.texts, self.labels, strict=True)


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def read_sentences(path: str | Path) -> list[str]:
    """Read one tweet per line, preserving commas inside quoted CSV fields."""
    texts: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as handle:
        for raw in handle:
            line = _strip_bom(raw).rstrip("\r\n")
            if not line:
                continue
            row = next(csv.reader([line]))
            texts.append(",".join(row).strip())
    return texts


def read_sentences_legacy(path: str | Path) -> list[str]:
    """Reproduce ``data_utils.ReadOpen`` sentence reconstruction.

    Each line is ``' '.join(line.strip().split(','))``. Quoted fields are not
    treated as CSV, so interior commas become spaces.
    """
    texts: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _strip_bom(raw).strip()
            if not line:
                continue
            texts.append(" ".join(line.split(",")))
    return texts


def read_labels(path: str | Path) -> list[int]:
    """Read one integer label per line (0 = non-sarcastic, 1 = sarcastic)."""
    labels: list[int] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _strip_bom(raw).strip()
            if not line:
                continue
            labels.append(int(line))
    return labels


def load_split(name: str, *, legacy: bool = False) -> DatasetSplit:
    """Load a named split from ``dataset/``.

    Parameters
    ----------
    name:
        ``train``, ``test``, or ``subtest``.
    legacy:
        If true, reconstruct tweet text the same way as ``ReadOpen``.
    """
    try:
        files = SPLIT_FILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SPLIT_FILES))
        raise ValueError(f"Unknown split {name!r}. Expected one of: {known}") from exc

    reader = read_sentences_legacy if legacy else read_sentences
    texts = reader(files["sentences"])
    labels = read_labels(files["labels"])
    if len(texts) != len(labels):
        raise ValueError(
            f"{name} has {len(texts)} sentences and {len(labels)} labels"
        )
    return DatasetSplit(name=name, texts=texts, labels=labels)


def load_all_splits(*, legacy: bool = False) -> dict[str, DatasetSplit]:
    return {name: load_split(name, legacy=legacy) for name in SPLIT_FILES}


def iter_labeled(texts: Sequence[str], labels: Sequence[int]) -> Iterable[tuple[str, int]]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")
    return zip(texts, labels, strict=True)
