"""Line-oriented loaders for ``dataset/*_sentence.csv`` and ``*_label.csv``.

The files are not headered CSV tables. See ``docs/dataset.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
SPLIT_NAMES = ("train", "test", "subtest")


def default_dataset_dir() -> Path:
    return DATASET_DIR


def strip_wrapping_quotes(line: str) -> str:
    """Drop a single wrapping ``"..."`` pair used when a tweet contains commas."""
    if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
        return line[1:-1]
    return line


def faithful_readopen_text(line: str) -> str:
    """Replay ``data_utils.ReadOpen`` text normalization (comma → space)."""
    return " ".join(line.strip().split(","))


def normalize_sentence(raw_line: str, *, faithful: bool = False) -> str:
    line = raw_line.rstrip("\r\n")
    if faithful:
        return faithful_readopen_text(line)
    return strip_wrapping_quotes(line)


def read_sentences(path: Path | str, *, faithful: bool = False) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [normalize_sentence(line, faithful=faithful) for line in handle]


def read_labels(path: Path | str) -> list[int]:
    path = Path(path)
    labels: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                labels.append(int(text))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: not an integer label: {raw!r}") from exc
    return labels


@dataclass(frozen=True)
class Split:
    name: str
    sentences: tuple[str, ...]
    labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return zip(self.sentences, self.labels)

    @property
    def n_positive(self) -> int:
        return sum(1 for label in self.labels if label == 1)

    @property
    def n_negative(self) -> int:
        return sum(1 for label in self.labels if label == 0)

    def labeled(self, value: int) -> Iterator[str]:
        for sentence, label in self:
            if label == value:
                yield sentence


def load_split(
    name: str,
    *,
    dataset_dir: Path | str | None = None,
    faithful: bool = False,
) -> Split:
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    root = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    sentences = read_sentences(root / f"{name}_sentence.csv", faithful=faithful)
    labels = read_labels(root / f"{name}_label.csv")
    if len(sentences) != len(labels):
        raise ValueError(
            f"{name}: {len(sentences)} sentences vs {len(labels)} labels"
        )
    return Split(name=name, sentences=tuple(sentences), labels=tuple(labels))


def load_all_splits(
    *,
    dataset_dir: Path | str | None = None,
    faithful: bool = False,
    names: Sequence[str] = SPLIT_NAMES,
) -> dict[str, Split]:
    return {
        name: load_split(name, dataset_dir=dataset_dir, faithful=faithful)
        for name in names
    }


def word_count(sentence: str) -> int:
    return len(sentence.split())


def iter_pairs(split: Split) -> Iterable[tuple[str, int]]:
    return zip(split.sentences, split.labels)
