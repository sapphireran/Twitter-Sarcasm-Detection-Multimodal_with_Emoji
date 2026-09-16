"""Load the course-project tweet/label CSVs without pandas.

Each split is a pair of files with one record per line:

* ``{split}_sentence.csv`` — raw tweet text. Some lines are wrapped in
  CSV quotes because the original dump stored commas inside tweets.
* ``{split}_label.csv`` — ``0`` (non-sarcastic) or ``1`` (sarcastic).

The 2023 ``ReadOpen`` helper in ``data_utils.py`` additionally splits each
line on commas and rejoins with spaces. That is a historical quirk of the
course dump (the files are *not* well-formed multi-column CSVs). This
module exposes both a faithful replay and a quote-aware reader.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

SPLITS: tuple[str, ...] = ("train", "test", "subtest")


@dataclass(frozen=True)
class TweetRecord:
    """One labelled tweet after line-level loading."""

    text: str
    label: int
    tokens: tuple[str, ...] | None = None

    @property
    def sarcastic(self) -> bool:
        return self.label == 1


@dataclass(frozen=True)
class Split:
    """Aligned sentence/label pair for one official split."""

    name: str
    texts: tuple[str, ...]
    labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self) -> Iterator[TweetRecord]:
        for text, label in zip(self.texts, self.labels):
            yield TweetRecord(text=text, label=label)

    def sarcastic_rate(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.labels) / len(self.labels)


def _read_label_file(path: Path) -> tuple[int, ...]:
    labels: list[int] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        labels.append(int(line))
    return tuple(labels)


def _read_sentence_file(path: Path, *, replay_readopen: bool) -> tuple[str, ...]:
    """Read one tweet per physical line.

    The files mix bare text with CSV-quoted rows. A csv.reader pass would
    swallow some of those quotes, which is closer to how a spreadsheet
    would open the dump. ``ReadOpen`` never used a CSV parser: it stripped
    the line, split on every comma, and joined the pieces with spaces.
    """

    texts: list[str] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if replay_readopen:
        for line in raw_lines:
            texts.append(" ".join(line.strip().split(",")))
        return tuple(texts)

    # Quote-aware fallback: keep a line even when it is not valid CSV.
    reader = csv.reader(raw_lines)
    for row, raw in zip(reader, raw_lines):
        if not row:
            texts.append(raw)
            continue
        texts.append(",".join(row).strip())
    # csv.reader stops if the last physical lines are empty; pad if needed.
    if len(texts) < len(raw_lines):
        texts.extend(raw_lines[len(texts) :])
    return tuple(texts)


def load_split(
    name: str,
    *,
    dataset_dir: Path | None = None,
    replay_readopen: bool = False,
) -> Split:
    """Load ``train``, ``test``, or ``subtest``.

    Parameters
    ----------
    name:
        Split name without the ``_sentence`` / ``_label`` suffix.
    dataset_dir:
        Override for tests that point at a fixture directory.
    replay_readopen:
        If true, apply the 2023 ``ReadOpen`` comma-join transform.
    """

    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    root = dataset_dir or DATASET_DIR
    texts = _read_sentence_file(root / f"{name}_sentence.csv", replay_readopen=replay_readopen)
    labels = _read_label_file(root / f"{name}_label.csv")
    if len(texts) != len(labels):
        raise ValueError(
            f"{name} alignment error: {len(texts)} sentences vs {len(labels)} labels"
        )
    return Split(name=name, texts=texts, labels=labels)


def iter_splits(
    names: Sequence[str] = SPLITS,
    **kwargs,
) -> Iterator[Split]:
    for name in names:
        yield load_split(name, **kwargs)


def split_summary(split: Split) -> dict[str, object]:
    """Counts and rates used by the dataset explorer."""

    counts = Counter(split.labels)
    n = len(split)
    lengths = [len(text) for text in split.texts]
    return {
        "name": split.name,
        "n": n,
        "n_non_sarcastic": counts.get(0, 0),
        "n_sarcastic": counts.get(1, 0),
        "sarcastic_rate": split.sarcastic_rate(),
        "mean_chars": (sum(lengths) / n) if n else 0.0,
        "median_chars": _median(lengths),
        "max_chars": max(lengths) if lengths else 0,
    }


def _median(values: Iterable[int]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0
