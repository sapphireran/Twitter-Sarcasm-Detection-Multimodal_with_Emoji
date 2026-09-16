"""Shared helpers for the documentation examples.

These modules are deliberately independent of TensorFlow, Gensim, NLTK, and
pandas so they run in a minimal environment. They re-implement the *ideas* from
``data_utils.py`` (tweet tokenization, emoji extraction, average pooling) without
changing the original 2023 training code.
"""

from .io import REPO_ROOT, load_split, iter_splits, Split
from .tokenize import tokenize_tweet, strip_hashtags, strip_urls
from .emoji import extract_emojis, is_emoji, EMOJI_RE
from .word2vec import load_word2vec_binary, cosine, nearest, lookup_emoji

__all__ = [
    "REPO_ROOT",
    "load_split",
    "iter_splits",
    "Split",
    "tokenize_tweet",
    "strip_hashtags",
    "strip_urls",
    "extract_emojis",
    "is_emoji",
    "EMOJI_RE",
    "load_word2vec_binary",
    "lookup_emoji",
    "cosine",
    "nearest",
]
