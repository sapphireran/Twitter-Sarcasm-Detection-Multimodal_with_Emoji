"""CSV readers that match data_utils.ReadOpen without pandas or NLTK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Split:
    name: str
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def pairs(self) -> Iterable[tuple[str, int]]:
        return zip(self.texts, self.labels)

    def positive_rate(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.labels) / len(self.labels)


def _flatten_commas(line: str) -> str:
    """Same flatten as ReadOpen: split on commas, re-join with spaces."""
    return " ".join(line.strip().split(","))


def read_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [_flatten_commas(line) for line in text.splitlines() if line.strip() != ""]


def read_labels(path: Path) -> list[int]:
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        labels.append(int(stripped))
    return labels


def load_split(sentence_path: Path, label_path: Path, name: str | None = None) -> Split:
    texts = read_sentences(sentence_path)
    labels = read_labels(label_path)
    if len(texts) != len(labels):
        raise ValueError(
            f"length mismatch: {sentence_path} has {len(texts)} rows, "
            f"{label_path} has {len(labels)}"
        )
    return Split(name=name or sentence_path.stem, texts=texts, labels=labels)


def load_all_splits(root: Path) -> dict[str, Split]:
    root = Path(root)
    mapping = {
        "train": ("train_sentence.csv", "train_label.csv"),
        "test": ("test_sentence.csv", "test_label.csv"),
        "subtest": ("subtest_sentence.csv", "subtest_label.csv"),
    }
    return {
        name: load_split(root / sent, root / lab, name=name)
        for name, (sent, lab) in mapping.items()
    }


def overlap_count(left: Split, right: Split) -> int:
    right_set = set(right.texts)
    return sum(1 for text in left.texts if text in right_set)
