"""Load the line-aligned tweet / label files under ``dataset/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import split_paths
from .tokenize import tokenize_tweet


@dataclass(frozen=True)
class Split:
    name: str
    texts: list[str]
    labels: list[int]
    tokens: list[list[str]]

    def __len__(self) -> int:
        return len(self.texts)

    def sarcastic_rate(self) -> float:
        if not self.labels:
            return 0.0
        return sum(self.labels) / len(self.labels)

    def label_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts


def _read_texts(path: Path) -> list[str]:
    # errors="replace" matches data_utils.ReadOpen.
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    texts: list[str] = []
    for line in lines:
        text = line.strip()
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1]
        texts.append(text)
    return texts


def _read_labels(path: Path) -> list[int]:
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        labels.append(int(stripped))
    return labels


def load_split(name: str, *, tokenize: bool = True) -> Split:
    sentence_path, label_path = split_paths(name)
    texts = _read_texts(sentence_path)
    labels = _read_labels(label_path)
    if len(texts) != len(labels):
        raise ValueError(
            f"{name}: {len(texts)} texts vs {len(labels)} labels "
            f"({sentence_path.name} / {label_path.name})"
        )
    tokens = [tokenize_tweet(text) for text in texts] if tokenize else []
    return Split(name=name, texts=texts, labels=labels, tokens=tokens)


def load_all_splits(*, tokenize: bool = True) -> dict[str, Split]:
    return {name: load_split(name, tokenize=tokenize) for name in ("train", "test", "subtest")}
