"""Self-contained examples for the 2023 UCPH sarcasm-detection project.

These modules reimplement the *ideas* in ``data_utils.py``,
``attention_layer.py``, and ``dl_model.py`` so they can be read and run
without TensorFlow, Gensim, NLTK, or the large GloVe Twitter vectors.
"""

from examples.attention import attention_forward
from examples.dataset_io import read_labels, read_open_replica, read_sentence_strings
from examples.embeddings import average_vector_per_sequence, build_embedding_matrix
from examples.emoji import EMOJI_RE, extract_emojis, has_emoji
from examples.tokenize import tokenize_tweet

__all__ = [
    "attention_forward",
    "average_vector_per_sequence",
    "build_embedding_matrix",
    "extract_emojis",
    "has_emoji",
    "EMOJI_RE",
    "read_labels",
    "read_open_replica",
    "read_sentence_strings",
    "tokenize_tweet",
]
