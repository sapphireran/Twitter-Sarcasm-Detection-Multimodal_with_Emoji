"""Dataset loaders that match the line-oriented CSV layout of this repo.

The original files are *not* RFC-4180 CSVs. Each line is one tweet (train /
test / subtest sentences) or one integer label. Quoted commas stay inside the
tweet string; ``data_utils.ReadOpen`` later replaces commas with spaces before
tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

SPLIT_NAMES: Tuple[str, ...] = ("train", "test", "subtest")


@dataclass(frozen=True)
class Split:
    """One labeled tweet collection."""

    name: str
    texts: List[str]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_sarcastic(self) -> int:
        return sum(1 for y in self.labels if y == 1)

    @property
    def n_literal(self) -> int:
        return sum(1 for y in self.labels if y == 0)

    def pairs(self) -> Iterator[Tuple[str, int]]:
        return zip(self.texts, self.labels)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def load_split(name: str, dataset_dir: Path | None = None) -> Split:
    """Load ``{name}_sentence.csv`` and ``{name}_label.csv``.

    Labels are stored as the strings ``0`` / ``1`` with no header. The
    sentence file keeps the original tweet, including surrounding quotes when
    the source export quoted a comma-containing line.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    root = dataset_dir or DATASET_DIR
    texts = _read_lines(root / f"{name}_sentence.csv")
    labels = [int(x.strip()) for x in _read_lines(root / f"{name}_label.csv")]
    if len(texts) != len(labels):
        raise ValueError(
            f"{name}: sentence/label length mismatch ({len(texts)} vs {len(labels)})"
        )
    return Split(name=name, texts=texts, labels=labels)


def iter_splits(dataset_dir: Path | None = None) -> Sequence[Split]:
    return tuple(load_split(name, dataset_dir=dataset_dir) for name in SPLIT_NAMES)
