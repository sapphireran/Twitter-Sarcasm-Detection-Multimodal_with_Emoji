"""Lightweight helpers for the personal multimodal Twitter sarcasm project.

This package is intentionally independent of TensorFlow, Gensim, and NLTK so
the documentation examples can run on a stock Python install. The original
course notebooks still use ``data_utils.py`` and ``dl_model.py``.
"""

from .attention import raffel_attention, softmax
from .emoji import EMOJI_RE, extract_emojis, has_emoji
from .heuristic import HeuristicResult, predict_sarcasm, score_tweet
from .io import DatasetSplit, load_split, read_labels, read_sentences
from .metrics import binary_metrics, confusion_counts
from .paths import DATASET_DIR, REPO_ROOT, SPLIT_FILES
from .stats import SplitStats, compute_split_stats
from .tokenize import tokenize_tweet

__all__ = [
    "DATASET_DIR",
    "EMOJI_RE",
    "HeuristicResult",
    "REPO_ROOT",
    "SPLIT_FILES",
    "SplitStats",
    "DatasetSplit",
    "binary_metrics",
    "compute_split_stats",
    "confusion_counts",
    "extract_emojis",
    "has_emoji",
    "load_split",
    "predict_sarcasm",
    "raffel_attention",
    "read_labels",
    "read_sentences",
    "score_tweet",
    "softmax",
    "tokenize_tweet",
]

__version__ = "0.2.0"
