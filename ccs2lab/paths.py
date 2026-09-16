"""Repository paths for the 2023 UCPH CCS2 sarcasm archive."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DOCS_DIR = ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "generated"
EXAMPLES_DIR = ROOT / "examples"
MODEL_DIR = ROOT / "model"
BASELINE_DIR = ROOT / "baseline_models"

SPLIT_FILES = {
    "train": (
        DATASET_DIR / "train_sentence.csv",
        DATASET_DIR / "train_label.csv",
    ),
    "test": (
        DATASET_DIR / "test_sentence.csv",
        DATASET_DIR / "test_label.csv",
    ),
    "subtest": (
        DATASET_DIR / "subtest_sentence.csv",
        DATASET_DIR / "subtest_label.csv",
    ),
}

EMOJI2VEC_300 = ROOT / "emoji2vec.bin"
EMOJI2VEC_TWITTER_200 = ROOT / "emoji2vec_twitter.bin"

# Named in the 2023 notebooks; never checked in.
GLOVE_TWITTER_BIN = ROOT / "glove.twitter.27B.200d.bin"
GLOVE_TWITTER_TXT = ROOT / "glove_tt.txt"

SAVED_MULTI = MODEL_DIR / "best_model_multi_modal"
SAVED_SINGLE = MODEL_DIR / "best_model_single_modal"
