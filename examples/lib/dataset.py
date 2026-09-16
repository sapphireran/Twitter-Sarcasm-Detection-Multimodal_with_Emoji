"""Load the checked-in train / test / subtest CSV pairs.

The original `ReadOpen` does not use a CSV parser. These helpers keep
the same line-aligned contract: one tweet per line, one integer label
per line, same row count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

SPLIT_NAMES = ("train", "test", "subtest")


class Split(NamedTuple):
    name: str
    sentences: List[str]
    labels: List[int]

    @property
    def n(self) -> int:
        return len(self.sentences)

    def pairs(self) -> List[Tuple[str, int]]:
        return list(zip(self.sentences, self.labels))


SPLITS = SPLIT_NAMES


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Keep blank-line policy aligned with the original notebooks:
    # empty lines are rare and would desynchronize labels, so drop them
    # only after confirming the sibling file matches.
    return [line.rstrip("\n") for line in text.splitlines() if line.strip() != ""]


def load_split(name: str, dataset_dir: Path = DATASET_DIR) -> Split:
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    sentences = _read_lines(dataset_dir / f"{name}_sentence.csv")
    raw_labels = _read_lines(dataset_dir / f"{name}_label.csv")
    labels = [int(row.split(",")[0]) for row in raw_labels]
    if len(sentences) != len(labels):
        raise ValueError(
            f"{name}: {len(sentences)} sentences vs {len(labels)} labels"
        )
    return Split(name, sentences, labels)


def load_all(dataset_dir: Path = DATASET_DIR) -> Dict[str, Split]:
    return {name: load_split(name, dataset_dir) for name in SPLIT_NAMES}
