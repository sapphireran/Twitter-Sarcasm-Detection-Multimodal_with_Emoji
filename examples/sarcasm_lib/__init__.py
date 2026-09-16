"""Helpers for the personal sarcasm-detection docs examples.

The package stays on the Python standard library plus NumPy so it can run
in environments that do not have TensorFlow, Gensim, NLTK, or scikit-learn.
It does not import the original ``data_utils`` / ``dl_model`` modules.
"""

from .attention import masked_temporal_attention
from .dataset import Split, load_all_splits, load_split
from .metrics import accuracy, binary_f1, confusion, precision, recall
from .tokenize import TweetishTokenizer, tokenize_tweet

__all__ = [
    "Split",
    "TweetishTokenizer",
    "accuracy",
    "binary_f1",
    "confusion",
    "load_all_splits",
    "load_split",
    "masked_temporal_attention",
    "precision",
    "recall",
    "tokenize_tweet",
]
