"""Load the project's sentence/label CSV splits without extra dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Split:
    """One labeled tweet split."""

    name: str
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def pairs(self) -> list[tuple[str, int]]:
        return list(zip(self.texts, self.labels, strict=True))


SPLIT_FILES = {
    "train": ("dataset/train_sentence.csv", "dataset/train_label.csv"),
    "test": ("dataset/test_sentence.csv", "dataset/test_label.csv"),
    "subtest": ("dataset/subtest_sentence.csv", "dataset/subtest_label.csv"),
}


def repo_root(start: Path | None = None) -> Path:
    """Walk upward until the dataset directory is found."""
    here = (start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents]:
        if (candidate / "dataset" / "train_sentence.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project dataset directory.")


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def load_split(name: str, root: Path | None = None) -> Split:
    """Load a named split (`train`, `test`, or `subtest`)."""
    if name not in SPLIT_FILES:
        known = ", ".join(sorted(SPLIT_FILES))
        raise KeyError(f"Unknown split {name!r}. Expected one of: {known}")

    root = root or repo_root()
    sentence_rel, label_rel = SPLIT_FILES[name]
    texts = _read_lines(root / sentence_rel)
    labels = [int(value) for value in _read_lines(root / label_rel) if value.strip() != ""]
    if len(texts) != len(labels):
        raise ValueError(
            f"{name} sentences ({len(texts)}) and labels ({len(labels)}) are different lengths"
        )
    return Split(name=name, texts=texts, labels=labels)


def load_all_splits(root: Path | None = None) -> dict[str, Split]:
    """Load train, test, and subtest."""
    return {name: load_split(name, root=root) for name in SPLIT_FILES}
