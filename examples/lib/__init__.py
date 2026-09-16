"""Self-contained helpers for the docs examples.

These modules re-implement the coursework ideas (tweet load, mean-pool
fusion, Raffel attention, lexical features) with the standard library
plus NumPy. They do not import `data_utils.py`, TensorFlow, gensim, or
nltk, so they run in a clone that is missing GloVe.
"""

from .attention_numpy import raffel_attention
from .dataset_io import Split, cue_stats, iter_labeled, load_split, split_sizes
from .embeddings import KeyedTable, average_rows, fuse_modalities
from .lexical_features import FEATURE_NAMES, lexical_feature_matrix, lexical_vector
from .logreg import LogisticRegressionGD
from .tweet_tokenize import tokenize_tweet

__all__ = [
    "FEATURE_NAMES",
    "KeyedTable",
    "LogisticRegressionGD",
    "Split",
    "average_rows",
    "cue_stats",
    "fuse_modalities",
    "iter_labeled",
    "lexical_feature_matrix",
    "lexical_vector",
    "load_split",
    "raffel_attention",
    "split_sizes",
    "tokenize_tweet",
]
