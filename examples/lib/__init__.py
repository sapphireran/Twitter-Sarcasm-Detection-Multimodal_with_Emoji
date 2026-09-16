"""Lightweight helpers for the personal sarcasm-detection examples.

These modules stay independent of TensorFlow, Gensim, NLTK, and the
``emoji`` package so they can run from a stock Python + NumPy environment.
They recreate the *ideas* of ``data_utils.py`` and ``attention_layer.py``
without requiring the 200-d GloVe Twitter dump used in the 2023 notebooks.
"""

from .attention import attention_forward, attention_weights
from .cues import SARCASM_HASHTAGS, extract_cues, rule_predict
from .dataset import DATASET_DIR, REPO_ROOT, load_split, split_summary
from .embeddings import (
    DummyKeyedVectors,
    average_vectors,
    build_padded_sequences,
    concatenate_modalities,
)
from .metrics import RECORDED_RESULTS, format_results_table, metric_bundle
from .tokenize import extract_emojis, is_emoji, tokenize_tweet

__all__ = [
    "DATASET_DIR",
    "REPO_ROOT",
    "RECORDED_RESULTS",
    "SARCASM_HASHTAGS",
    "DummyKeyedVectors",
    "attention_forward",
    "attention_weights",
    "average_vectors",
    "build_padded_sequences",
    "concatenate_modalities",
    "extract_cues",
    "extract_emojis",
    "format_results_table",
    "is_emoji",
    "load_split",
    "metric_bundle",
    "rule_predict",
    "split_summary",
    "tokenize_tweet",
]
