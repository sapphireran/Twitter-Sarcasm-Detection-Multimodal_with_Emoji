"""Stdlib helpers for the personal sarcasm-detection walkthroughs."""

from .attention import temporal_attention
from .embeddings import average_vectors, toy_table
from .features import HASHTAG_LEAK_PATTERNS, extract_features, has_leak_hashtag
from .io import Split, load_split, repo_root
from .metrics import accuracy, confusion, f1, precision, recall
from .naive_bayes import BernoulliNB
from .tokenize import is_emoji_token, is_hashtag_token, tokenize_tweet

__all__ = [
    "BernoulliNB",
    "HASHTAG_LEAK_PATTERNS",
    "Split",
    "accuracy",
    "average_vectors",
    "confusion",
    "extract_features",
    "f1",
    "has_leak_hashtag",
    "load_split",
    "precision",
    "recall",
    "repo_root",
    "temporal_attention",
    "is_emoji_token",
    "is_hashtag_token",
    "tokenize_tweet",
    "toy_table",
]
