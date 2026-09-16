"""Resolve committed dataset files from any working directory."""

from pathlib import Path

# examples/sarcasm_lab/paths.py → repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

SPLIT_FILES = {
    "train": ("train_sentence.csv", "train_label.csv"),
    "test": ("test_sentence.csv", "test_label.csv"),
    "subtest": ("subtest_sentence.csv", "subtest_label.csv"),
}


def split_paths(name: str) -> tuple[Path, Path]:
    try:
        sentences, labels = SPLIT_FILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown split {name!r}; choose {sorted(SPLIT_FILES)}") from exc
    return DATASET_DIR / sentences, DATASET_DIR / labels
