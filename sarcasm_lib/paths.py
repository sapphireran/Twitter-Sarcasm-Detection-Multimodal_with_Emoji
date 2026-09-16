"""Repository-relative paths used by docs and examples."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MODEL_DIR = REPO_ROOT / "model"
BASELINE_DIR = REPO_ROOT / "baseline_models"

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

EMOJI2VEC_TWITTER = REPO_ROOT / "emoji2vec_twitter.bin"
EMOJI2VEC = REPO_ROOT / "emoji2vec.bin"
