"""Read the course dataset without pandas.

Each split is a pair of line-aligned files:

* ``{split}_sentence.csv`` — one tweet per line (sometimes quoted)
* ``{split}_label.csv`` — ``0`` (non-sarcastic) or ``1`` (sarcastic)

The original ``data_utils.ReadOpen`` joins commas inside a line and then
runs NLTK's ``TweetTokenizer``. That is fine for the 2023 notebooks, but
it is heavy for docs/examples. This module keeps the raw tweet text and
only strips wrapping quotes so inspection scripts can show real examples.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Iterator, Sequence

from .paths import SPLIT_FILES


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    split: str
    index: int

    @property
    def sarcastic(self) -> bool:
        return self.label == 1


@dataclass(frozen=True)
class Split:
    name: str
    texts: tuple[str, ...]
    labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self) -> Iterator[Example]:
        return iter_examples(self)

    @property
    def n_sarcastic(self) -> int:
        return sum(self.labels)

    @property
    def n_literal(self) -> int:
        return len(self) - self.n_sarcastic


def _strip_wrapping_quotes(line: str) -> str:
    text = line.rstrip("\n\r")
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def _read_lines(path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [_strip_wrapping_quotes(line) for line in handle]


def _read_labels(path) -> list[int]:
    labels: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            labels.append(int(stripped))
    return labels


def load_split(name: str) -> Split:
    """Load ``train``, ``test``, or ``subtest`` from ``dataset/``."""
    if name not in SPLIT_FILES:
        known = ", ".join(sorted(SPLIT_FILES))
        raise ValueError(f"unknown split {name!r}; expected one of {known}")
    files = SPLIT_FILES[name]
    texts = _read_lines(files["sentences"])
    labels = _read_labels(files["labels"])
    if len(texts) != len(labels):
        raise ValueError(
            f"{name} sentence/label mismatch: {len(texts)} texts vs {len(labels)} labels"
        )
    return Split(name=name, texts=tuple(texts), labels=tuple(labels))


def iter_examples(split: Split) -> Iterator[Example]:
    for index, (text, label) in enumerate(zip(split.texts, split.labels)):
        yield Example(text=text, label=label, split=split.name, index=index)


def summarize_split(split: Split) -> dict:
    """Return JSON-serializable counts used by the inspect example."""
    n = len(split)
    sarcastic = split.n_sarcastic
    token_lens = [len(text.split()) for text in split.texts]
    char_lens = [len(text) for text in split.texts]
    return {
        "split": split.name,
        "n": n,
        "sarcastic": sarcastic,
        "literal": n - sarcastic,
        "sarcastic_rate": sarcastic / n if n else 0.0,
        "token_len": _minmax_mean(token_lens),
        "char_len": _minmax_mean(char_lens),
        "label_counts": dict(Counter(split.labels)),
    }


def sample_split(split: Split, n: int, seed: int = 2023) -> Split:
    """Shuffle and take ``n`` rows. Train CSVs are label-sorted in places,
    so slicing ``[:n]`` is a biased subset — always sample instead.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    count = min(n, len(split))
    order = list(range(len(split)))
    Random(seed).shuffle(order)
    picked = order[:count]
    return Split(
        name=split.name,
        texts=tuple(split.texts[i] for i in picked),
        labels=tuple(split.labels[i] for i in picked),
    )


def _minmax_mean(values: Sequence[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "median": median,
    }
