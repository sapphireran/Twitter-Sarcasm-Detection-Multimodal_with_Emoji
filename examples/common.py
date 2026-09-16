"""Shared helpers for the personal course-project examples.

These loaders stay on the Python standard library plus NumPy so they run in a
minimal environment. They deliberately reimplement the *observable* behavior of
``data_utils.ReadOpen`` without importing TensorFlow, Keras, NLTK, or pandas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
EMOJI2VEC_TWITTER = REPO_ROOT / "emoji2vec_twitter.bin"
EMOJI2VEC_ORIGINAL = REPO_ROOT / "emoji2vec.bin"

SPLITS = ("train", "test", "subtest")

# Broad but not exhaustive. Good enough to describe this snapshot; not a
# complete Unicode emoji property implementation.
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF"
    r"\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]"
)
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"<user>|@\w+", re.IGNORECASE)

# Hashtags that were historically used as distant-supervision sarcasm labels.
SARCASM_CUE_TAGS = (
    "#not",
    "#sarcasm",
    "#sarcastic",
    "#sarcastictweet",
    "#yeahright",
    "#yahright",
    "#notsarcastic",
)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def read_label_file(path: Path) -> np.ndarray:
    """Read one integer label per non-empty line."""
    labels: List[int] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            labels.append(int(line))
    return np.asarray(labels, dtype=np.int64)


def read_sentence_file(path: Path) -> List[str]:
    """Read one tweet per physical line, preserving inner whitespace."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n\r") for line in handle]


def load_split(split: str) -> Tuple[List[str], np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {SPLITS}.")
    sentences = read_sentence_file(DATASET_DIR / f"{split}_sentence.csv")
    labels = read_label_file(DATASET_DIR / f"{split}_label.csv")
    if len(sentences) != len(labels):
        raise ValueError(
            f"{split}: {len(sentences)} sentences vs {len(labels)} labels"
        )
    return sentences, labels


def collapse_commas(text: str) -> str:
    """Match ``data_utils.ReadOpen``: commas become spaces, then re-join."""
    return " ".join(text.strip().split(","))


def simple_tweet_tokens(text: str) -> List[str]:
    """A small TweetTokenizer stand-in.

    Keeps hashtags, @mentions, <user>, urls, and emoji clusters as tokens.
    This is for documentation examples, not a drop-in NLTK replacement.
    """
    cleaned = collapse_commas(text)
    pattern = re.compile(
        r"https?://\S+"
        r"|<user>"
        r"|[@#][\w']+"
        r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF"
        r"\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]"
        r"|[\w']+"
        r"|[^\s\w]",
        re.UNICODE,
    )
    return [tok.lower() for tok in pattern.findall(cleaned) if tok.strip()]


def whitespace_tokens(text: str) -> List[str]:
    return collapse_commas(text).lower().split()


def iter_hashtags(text: str) -> Iterator[str]:
    for match in HASHTAG_RE.findall(text.lower()):
        yield match


def has_emoji(text: str) -> bool:
    return EMOJI_RE.search(text) is not None


def has_mention(text: str) -> bool:
    return MENTION_RE.search(text) is not None


def sarcasm_cue_tags(text: str) -> List[str]:
    tags = set(iter_hashtags(text))
    return [tag for tag in SARCASM_CUE_TAGS if tag in tags]


def majority_label(labels: Sequence[int]) -> int:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return int(values[int(np.argmax(counts))])


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def binary_scores(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Accuracy / precision / recall / F1 for the sarcastic class (label 1)."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {
        "n": int(y_true_arr.size),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def format_scores(name: str, scores: dict) -> str:
    return (
        f"{name:18s}  n={scores['n']:5d}  "
        f"acc={scores['accuracy']:.3f}  "
        f"p={scores['precision']:.3f}  "
        f"r={scores['recall']:.3f}  "
        f"f1={scores['f1']:.3f}"
    )


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def load_word2vec_binary(path: Path, max_words: int | None = None) -> dict:
    """Read a little-endian C-format word2vec / emoji2vec binary.

    Returns a dict with ``vocab_size``, ``vector_size``, ``tokens`` (list),
    and ``vectors`` (float32 array of shape ``[n_loaded, vector_size]``).
    """
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="replace").strip()
        vocab_size_s, vector_size_s = header.split()
        vocab_size = int(vocab_size_s)
        vector_size = int(vector_size_s)
        n_load = vocab_size if max_words is None else min(vocab_size, max_words)
        tokens: List[str] = []
        vectors = np.zeros((n_load, vector_size), dtype=np.float32)
        for i in range(n_load):
            chars: List[bytes] = []
            while True:
                ch = handle.read(1)
                if not ch:
                    raise EOFError(f"Unexpected EOF in {path} at word {i}")
                if ch == b" ":
                    break
                if ch != b"\n":
                    chars.append(ch)
            token = b"".join(chars).decode("utf-8", errors="replace")
            raw = handle.read(4 * vector_size)
            if len(raw) != 4 * vector_size:
                raise EOFError(f"Truncated vector for {token!r} in {path}")
            tokens.append(token)
            vectors[i] = np.frombuffer(raw, dtype=np.float32)
        return {
            "path": str(path),
            "vocab_size": vocab_size,
            "vector_size": vector_size,
            "tokens": tokens,
            "vectors": vectors,
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def nearest(query: np.ndarray, matrix: np.ndarray, k: int = 8) -> List[int]:
    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        return []
    row_norm = np.linalg.norm(matrix, axis=1)
    safe = np.where(row_norm == 0.0, 1.0, row_norm)
    sims = matrix.dot(query) / (safe * q_norm)
    sims = np.where(row_norm == 0.0, -np.inf, sims)
    k = min(k, matrix.shape[0])
    # argpartition then sort the k candidates
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-sims[idx])]]


def take(iterable: Iterable, n: int) -> List:
    out = []
    for i, item in enumerate(iterable):
        if i >= n:
            break
        out.append(item)
    return out
