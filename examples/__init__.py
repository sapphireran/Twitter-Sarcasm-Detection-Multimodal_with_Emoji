"""Lightweight, numpy-only companions to the 2023 sarcasm-detection pipeline.

These modules reimplement the *math* in ``data_utils.AverageVectorPerTweet``,
``data_utils.AverageVectorPerEmoji``, and ``attention_layer.Attention`` without
Gensim, NLTK, Keras, or the GloVe dump. Import them from the repo root::

    python3 -m examples.run_all
"""

from .toy_corpus import TOY_LABELS, TOY_TWEETS, illustrative_pair

__all__ = ["TOY_LABELS", "TOY_TWEETS", "illustrative_pair"]
