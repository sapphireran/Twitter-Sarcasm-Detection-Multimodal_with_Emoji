"""Dependency-light stand-in for the 2023 ``data_utils`` / attention math.

The original project imports NLTK, gensim, Keras, and the ``emoji`` package.
Those are the right tools for replaying the notebooks. They are the wrong
tools for reading the repository on a laptop that only has NumPy.

This module keeps the *shapes* and the *formulas* of the 2023 code:

* tweet-like tokenization (hashtags, user tokens, emoji stay whole)
* optional comma-join that mimics ``ReadOpen``
* mean pooling into a word view and a concatenated emoji view
* Raffel feed-forward attention (tanh score, masked softmax, weighted sum)

It deliberately does **not** load GloVe or TensorFlow. Embedding tables here
are plain ``dict[str, np.ndarray]``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# Hashtags, <user>, @handles, url-ish tokens, remaining words, leftover chars.
# Emoji are matched as single Unicode code points in the pictographic blocks
# plus a few dingbats. This is not a grapheme-cluster parser; it is close
# enough to NLTK TweetTokenizer for the examples.
_TOKEN_RE = re.compile(
    r"<user>"
    r"|https?://\S+"
    r"|#\w+"
    r"|@\w+"
    r"|[\w']+"
    r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U00002600-\U000026FF]"
    r"|[^\s\w]",
    re.UNICODE,
)

CUE_HASHTAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#irony",
        "#ironic",
        "#yeahright",
    }
)

EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2700, 0x27BF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x2600, 0x26FF),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def tokenize(text: str) -> list[str]:
    """Lowercase tweet-like tokens. Keeps ``#not`` and emoji as atoms."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def mimic_readopen_commas(text: str) -> str:
    """Reproduce ``' '.join(line.strip().split(','))`` from ``ReadOpen``."""
    return " ".join(text.strip().split(","))


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def read_sentence_label_pair(
    sentence_path: str | Path,
    label_path: str | Path,
    *,
    join_commas: bool = False,
) -> tuple[list[list[str]], np.ndarray]:
    """Load an aligned ``*_sentence.csv`` / ``*_label.csv`` pair.

    ``join_commas=True`` applies the 2023 ``ReadOpen`` comma smash before
    tokenization. Default is to keep commas (they become their own token).
    """
    sentences = Path(sentence_path).read_text(encoding="utf-8", errors="replace").splitlines()
    labels_raw = Path(label_path).read_text(encoding="utf-8", errors="replace").splitlines()
    labels = [int(x.strip()) for x in labels_raw if x.strip() != ""]
    if len(sentences) != len(labels):
        raise ValueError(
            f"alignment error: {sentence_path} has {len(sentences)} lines, "
            f"{label_path} has {len(labels)} labels"
        )
    docs: list[list[str]] = []
    for line in sentences:
        text = strip_wrapping_quotes(line)
        if join_commas:
            text = mimic_readopen_commas(text)
        docs.append(tokenize(text))
    return docs, np.asarray(labels, dtype=np.int64)


def is_emoji_token(token: str) -> bool:
    if not token:
        return False
    # Multi-codepoint tokens are emoji if every char is in an emoji block.
    return all(any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES) for ch in token)


def has_cue_hashtag(tokens: Sequence[str]) -> bool:
    return any(tok in CUE_HASHTAGS for tok in tokens)


def mean_pool(
    tokens: Sequence[str],
    table: Mapping[str, np.ndarray],
    width: int,
) -> np.ndarray:
    """Average rows that hit ``table``. Zeros if nothing hit.

    Same contract as ``AverageVectorPerTweet`` / ``AverageVectorPerEmoji``.
    """
    rows = [np.asarray(table[tok], dtype=np.float64) for tok in tokens if tok in table]
    if not rows:
        return np.zeros((width,), dtype=np.float64)
    stacked = np.stack(rows, axis=0)
    if stacked.shape[1] != width:
        raise ValueError(f"embedding width {stacked.shape[1]} != {width}")
    return stacked.mean(axis=0)


def pooled_views(
    docs: Sequence[Sequence[str]],
    word_table: Mapping[str, np.ndarray],
    emoji_table: Mapping[str, np.ndarray],
    *,
    word_width: int = 200,
    emoji_width: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N, word_width)`` and ``(N, word_width + emoji_width)``."""
    word_rows = [mean_pool(doc, word_table, word_width) for doc in docs]
    emoji_rows = [mean_pool(doc, emoji_table, emoji_width) for doc in docs]
    x_word = np.stack(word_rows, axis=0)
    x_emoji = np.stack(emoji_rows, axis=0)
    x_multi = np.concatenate([x_word, x_emoji], axis=1)
    return x_word, x_multi


def build_embed_table(
    docs: Sequence[Sequence[str]],
    word_table: Mapping[str, np.ndarray],
    emoji_table: Mapping[str, np.ndarray],
    *,
    width: int = 200,
    use_emoji: bool = True,
) -> tuple[dict[str, int], np.ndarray]:
    """Train-time embedding matrix keyed by token, sized by **vocab**.

    ``data_utils.Preprocess`` sizes the matrix by tweet *count*. This
    function sizes it by ``max(index) + 1`` so tiny fixtures do not crash.
    Index 0 is padding (zeros).
    """
    vocab: dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1
    matrix = np.zeros((len(vocab) + 1, width), dtype=np.float64)
    for tok, idx in vocab.items():
        if tok in word_table:
            matrix[idx] = np.asarray(word_table[tok], dtype=np.float64)
        elif use_emoji and is_emoji_token(tok) and tok in emoji_table:
            matrix[idx] = np.asarray(emoji_table[tok], dtype=np.float64)
    return vocab, matrix


def docs_to_padded(
    docs: Sequence[Sequence[str]],
    vocab: Mapping[str, int],
    maxlen: int | None = None,
) -> np.ndarray:
    ids = [[vocab.get(tok, 0) for tok in doc] for doc in docs]
    if maxlen is None:
        maxlen = max((len(row) for row in ids), default=0)
    padded = np.zeros((len(ids), maxlen), dtype=np.int64)
    for i, row in enumerate(ids):
        clipped = row[:maxlen]
        padded[i, : len(clipped)] = clipped
    return padded


def raffel_attention(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Raffel attention matching ``attention_layer.Attention.call``.

    Parameters
    ----------
    x:
        ``(batch, steps, features)``
    weight:
        ``(features,)``  — the Keras ``W``
    bias:
        optional ``(steps,)``  — the Keras ``b``, added *before* tanh
    mask:
        optional ``(batch, steps)`` of 0/1
    epsilon:
        added to the softmax denominator, same motivation as Keras ``K.epsilon()``

    Returns
    -------
    context:
        ``(batch, features)``
    weights:
        ``(batch, steps)``
    """
    if x.ndim != 3:
        raise ValueError(f"expected (batch, steps, features), got {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"W shape {weight.shape} != ({x.shape[-1]},)")
    scores = np.tensordot(x, weight, axes=([-1], [0]))
    if bias is not None:
        if bias.shape != (x.shape[1],):
            raise ValueError(f"bias shape {bias.shape} != ({x.shape[1]},)")
        scores = scores + bias
    scores = np.tanh(scores)
    unnormalized = np.exp(scores)
    if mask is not None:
        unnormalized = unnormalized * mask.astype(unnormalized.dtype)
    denom = unnormalized.sum(axis=1, keepdims=True) + epsilon
    weights = unnormalized / denom
    context = (x * weights[..., None]).sum(axis=1)
    return context, weights


def cue_emoji_features(docs: Sequence[Sequence[str]]) -> np.ndarray:
    """Four hand features: cue hashtag, any emoji, token count, elongated 'love'."""
    rows = []
    for doc in docs:
        cue = float(has_cue_hashtag(doc))
        emo = float(any(is_emoji_token(t) for t in doc))
        length = float(len(doc))
        elong = float(any(re.fullmatch(r"lo+ve+", t) for t in doc))
        rows.append([cue, emo, length, elong])
    return np.asarray(rows, dtype=np.float64)


def nearest_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Tiny baseline used in examples: class-mean Euclidean assignment."""
    classes = np.unique(y_train)
    centroids = np.stack([x_train[y_train == c].mean(axis=0) for c in classes], axis=0)
    # (N, C) distances
    diffs = x_test[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diffs * diffs).sum(axis=2))
    return classes[dist.argmin(axis=1)]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("empty arrays")
    return float((y_true == y_pred).mean())


def write_aligned_csv(sentence_path: Path, label_path: Path, rows: Iterable[tuple[str, int]]) -> None:
    """Helper for building tiny fixtures without a real CSV library."""
    sentence_path.parent.mkdir(parents=True, exist_ok=True)
    with sentence_path.open("w", encoding="utf-8", newline="") as sf, label_path.open(
        "w", encoding="utf-8", newline=""
    ) as lf:
        sent_w = csv.writer(sf, lineterminator="\n")
        for text, lab in rows:
            # Always quote so commas in the tweet survive a naive split.
            sent_w.writerow([text])
            lf.write(f"{int(lab)}\n")
