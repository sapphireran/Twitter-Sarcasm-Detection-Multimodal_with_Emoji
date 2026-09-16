"""Dependency-light pieces that mirror the 2023 sarcasm pipeline."""

from .attention_numpy import attention_forward
from .dataset_io import Split, load_split, load_all_splits
from .emoji_extract import extract_emojis, has_high_codepoint, is_emoji_char
from .lexical_features import FEATURE_NAMES, featurize, featurize_corpus
from .logistic import LogisticBinary
from .mean_pool import average_rows, concat_modalities
from .tweet_tokenize import tokenize_tweet

__all__ = [
    "FEATURE_NAMES",
    "LogisticBinary",
    "Split",
    "attention_forward",
    "average_rows",
    "concat_modalities",
    "extract_emojis",
    "featurize",
    "featurize_corpus",
    "has_high_codepoint",
    "is_emoji_char",
    "load_all_splits",
    "load_split",
    "tokenize_tweet",
]
