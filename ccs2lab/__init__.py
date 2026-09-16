"""CCS2 2023 sarcasm-detection archive lab.

NumPy + stdlib only. The original notebooks still need TensorFlow,
gensim, NLTK, and the missing GloVe Twitter file.
"""

from ccs2lab.attention import raffel_attention
from ccs2lab.cues import CueProfile, EXPLICIT_CUES, profile_text
from ccs2lab.metrics import BinaryScores, binary_scores, wilson_interval
from ccs2lab.paths import ROOT, DATASET_DIR
from ccs2lab.recorded import RECORDED_ROWS
from ccs2lab.splits import SplitBundle, load_bundle
from ccs2lab.tokenize import tokenize_tweet

__all__ = [
    "BinaryScores",
    "CueProfile",
    "DATASET_DIR",
    "EXPLICIT_CUES",
    "RECORDED_ROWS",
    "ROOT",
    "SplitBundle",
    "binary_scores",
    "load_bundle",
    "profile_text",
    "raffel_attention",
    "tokenize_tweet",
    "wilson_interval",
]

__version__ = "0.2.0"
