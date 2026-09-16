"""Lightweight helpers for the personal sarcasm-detection examples.

The original 2023 notebooks depend on TensorFlow, Gensim, NLTK, and a
~1.2 GB GloVe dump that is not checked into this repository. This package
reimplements the pieces that are useful for documentation and local demos:

* dataset IO that does not need pandas
* a tweet-oriented tokenizer that does not need NLTK
* lexical sarcasm-cue features
* the Raffel-style attention pooling used by ``attention_layer.py``
* classification metrics used in the course report

Nothing here talks to social-media APIs. All examples read the local
``dataset/`` CSVs that already live in the repo.
"""

from .attention import attention_pool, softmax
from .baseline import CueLogistic, LexiconBaseline
from .cues import CUE_HASHTAGS, extract_cue_features, feature_names
from .dataset import Split, iter_examples, load_split, summarize_split
from .metrics import binary_metrics, format_metrics
from .paths import DATASET_DIR, REPO_ROOT
from .tokenize import tokenize_tweet

__all__ = [
    "CUE_HASHTAGS",
    "CueLogistic",
    "LexiconBaseline",
    "Split",
    "attention_pool",
    "binary_metrics",
    "extract_cue_features",
    "feature_names",
    "format_metrics",
    "iter_examples",
    "load_split",
    "softmax",
    "summarize_split",
    "tokenize_tweet",
    "DATASET_DIR",
    "REPO_ROOT",
]

__version__ = "0.2.0"
