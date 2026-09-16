"""Read the parallel sentence / label CSVs used by the coursework."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LABEL_RE = re.compile(r"^[01]\s*$")

# Broad but cheap emoji detector for dataset reports. It is not the
# same as the `emoji` package used in data_utils.Preprocess.
EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002700-\U000027bf"
    "\U0001f900-\U0001f9ff"
    "\U00002600-\U000026ff"
    "\U0001fa70-\U0001faff"
    "]+",
    flags=re.UNICODE,
)

SARC_TAG_RE = re.compile(
    r"#?(sarcasm|sarcastic(?:tweet)?|not|yeahright)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Split:
    name: str
    sentences: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.sentences)

    def labeled(self) -> Iterator[tuple[str, int]]:
        return zip(self.sentences, self.labels)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return _repo_root() / "dataset"


def read_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Keep blank-stripped lines but preserve internal whitespace.
    return [line.rstrip("\n") for line in text.splitlines() if line.strip()]


def read_labels(path: Path) -> list[int]:
    labels: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not LABEL_RE.match(line):
            raise ValueError(f"expected 0/1 label in {path}, got {raw!r}")
        labels.append(int(line))
    return labels


def load_split(name: str, dataset_dir: Path | None = None) -> Split:
    root = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    sentences = read_sentences(root / f"{name}_sentence.csv")
    labels = read_labels(root / f"{name}_label.csv")
    if len(sentences) != len(labels):
        raise ValueError(
            f"{name}: {len(sentences)} sentences vs {len(labels)} labels"
        )
    return Split(name=name, sentences=sentences, labels=labels)


def iter_labeled(
    names: tuple[str, ...] = ("train", "test", "subtest"),
    dataset_dir: Path | None = None,
) -> Iterator[tuple[str, str, int]]:
    for name in names:
        split = load_split(name, dataset_dir)
        for sentence, label in split.labeled():
            yield name, sentence, label


def split_sizes(split: Split) -> dict[str, int]:
    counts = Counter(split.labels)
    return {
        "rows": len(split),
        "literal": int(counts.get(0, 0)),
        "sarcastic": int(counts.get(1, 0)),
    }


def cue_stats(split: Split) -> dict[str, int | str]:
    """Surface-level counts used in docs/dataset.md."""
    has_hash = has_emoji = has_user = has_sarc = 0
    hashtag_counter: Counter[str] = Counter()
    token_lengths: list[int] = []
    char_lengths: list[int] = []
    for sentence in split.sentences:
        token_lengths.append(len(sentence.split()))
        char_lengths.append(len(sentence))
        if "#" in sentence:
            has_hash += 1
        if EMOJI_RE.search(sentence):
            has_emoji += 1
        lowered = sentence.lower()
        if "<user>" in lowered or "@" in sentence:
            has_user += 1
        if SARC_TAG_RE.search(sentence):
            has_sarc += 1
        hashtag_counter.update(re.findall(r"#\w+", lowered))
    return {
        "with_hashtag": has_hash,
        "with_emoji": has_emoji,
        "with_user": has_user,
        "with_sarc_tag": has_sarc,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "mean_tokens_x100": int(
            round(100 * sum(token_lengths) / len(token_lengths))
        )
        if token_lengths
        else 0,
        "min_chars": min(char_lengths) if char_lengths else 0,
        "max_chars": max(char_lengths) if char_lengths else 0,
        "top_hashtag": hashtag_counter.most_common(1)[0][0]
        if hashtag_counter
        else "",
        "top_hashtag_count": hashtag_counter.most_common(1)[0][1]
        if hashtag_counter
        else 0,
    }


def top_hashtags(split: Split, n: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for sentence in split.sentences:
        counter.update(re.findall(r"#\w+", sentence.lower()))
    return counter.most_common(n)
