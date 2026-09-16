"""Load the headerless sentence/label CSV pairs shipped in dataset/."""

from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "test", "subtest")


def repo_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def split_paths(name: str, root: Path | None = None) -> tuple[Path, Path]:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    base = (root or repo_root()) / "dataset"
    return base / f"{name}_sentence.csv", base / f"{name}_label.csv"


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def load_split(name: str, root: Path | None = None) -> list[tuple[str, int]]:
    """Pair sentence line i with label line i.

    Unlike ``data_utils.ReadOpen`` this does not rewrite commas and does
    not shuffle. Labels must be the integers 0 or 1.
    """
    sentence_path, label_path = split_paths(name, root)
    sentences = _read_lines(sentence_path)
    raw_labels = [line.strip() for line in _read_lines(label_path)]
    labels = [item for item in raw_labels if item != ""]
    if len(sentences) != len(labels):
        raise ValueError(
            f"{name}: {len(sentences)} sentences vs {len(labels)} labels "
            f"({sentence_path.name} / {label_path.name})"
        )
    paired: list[tuple[str, int]] = []
    for line_no, (tweet, label_text) in enumerate(zip(sentences, labels), start=1):
        if label_text not in {"0", "1"}:
            raise ValueError(f"{label_path}:{line_no}: label {label_text!r} is not 0 or 1")
        paired.append((tweet, int(label_text)))
    return paired


def class_counts(rows: list[tuple[str, int]]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for _, label in rows:
        counts[label] += 1
    return counts
