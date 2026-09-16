"""Small library used by the personal sarcasm-detection examples.

Nothing here loads GloVe, emoji2vec, TensorFlow, or NLTK. The original
training path still lives in ``data_utils.py`` / ``dl_model.py``.
"""

from .attention import raffel_attention, softmax
from .cues import CueFeatures, extract_cues
from .io import Split, load_split, load_all_splits
from .metrics import binary_metrics, format_metrics
from .naive_bayes import MultinomialNB
from .paths import DATASET_DIR, REPO_ROOT
from .sgd_logreg import SGDLogisticRegression
from .tokenize import tokenize_tweet
from .vectorize import CountVectorizer

__all__ = [
    "CueFeatures",
    "DATASET_DIR",
    "MultinomialNB",
    "REPO_ROOT",
    "SGDLogisticRegression",
    "Split",
    "binary_metrics",
    "extract_cues",
    "format_metrics",
    "load_all_splits",
    "load_split",
    "raffel_attention",
    "softmax",
    "tokenize_tweet",
    "CountVectorizer",
]
