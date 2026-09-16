"""Repository-relative paths so examples work from any cwd."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = REPO_ROOT / "examples"
BASELINE_DIR = REPO_ROOT / "baseline_models"
MODEL_DIR = REPO_ROOT / "model"

SPLIT_FILES = {
    "train": {
        "sentences": DATASET_DIR / "train_sentence.csv",
        "labels": DATASET_DIR / "train_label.csv",
    },
    "test": {
        "sentences": DATASET_DIR / "test_sentence.csv",
        "labels": DATASET_DIR / "test_label.csv",
    },
    "subtest": {
        "sentences": DATASET_DIR / "subtest_sentence.csv",
        "labels": DATASET_DIR / "subtest_label.csv",
    },
}

REPORTED_RESULTS_PATH = EXAMPLES_DIR / "reported_results.json"
