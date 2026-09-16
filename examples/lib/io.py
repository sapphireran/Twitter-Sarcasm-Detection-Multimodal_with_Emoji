"""Load the local CSV splits without pandas.

The sentence files are not reliable RFC 4180: a tweet that contains commas may
be stored as a quoted line. We keep the same rule as ``data_utils.ReadOpen`` —
join comma-separated pieces back into one string — and require a 1:1 alignment
with the label file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Split:
    name: str
    texts: Tuple[str, ...]
    labels: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def positive_rate(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.labels) / len(self.labels)


def repo_root() -> Path:
    """Return the repository root even if the cwd is ``examples/`` or ``tests/``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "dataset" / "train_sentence.csv").exists():
            return parent
    raise FileNotFoundError("Could not locate dataset/train_sentence.csv from " + str(here))


def _read_sentences(path: Path) -> List[str]:
    # errors="replace" matches data_utils.ReadOpen.
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    texts: List[str] = []
    for line in raw_lines:
        # Undo the accidental CSV split used when the original dump was written.
        sentence = " ".join(piece.strip() for piece in line.strip().split(","))
        if sentence.startswith('"') and sentence.endswith('"') and len(sentence) >= 2:
            sentence = sentence[1:-1]
        texts.append(sentence)
    return texts


def _read_labels(path: Path) -> List[int]:
    labels: List[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        labels.append(int(stripped))
    return labels


def load_split(name: str, root: Path | None = None) -> Split:
    """Load ``dataset/{name}_sentence.csv`` + ``{name}_label.csv``."""
    if name not in {"train", "test", "subtest"}:
        raise ValueError(f"unknown split {name!r}; expected train, test, or subtest")
    base = (root or repo_root()) / "dataset"
    texts = _read_sentences(base / f"{name}_sentence.csv")
    labels = _read_labels(base / f"{name}_label.csv")
    if len(texts) != len(labels):
        raise ValueError(
            f"{name}: {len(texts)} sentences vs {len(labels)} labels — files are not aligned"
        )
    return Split(name=name, texts=tuple(texts), labels=tuple(labels))


def load_all(root: Path | None = None) -> Sequence[Split]:
    return tuple(load_split(name, root=root) for name in ("train", "test", "subtest"))
