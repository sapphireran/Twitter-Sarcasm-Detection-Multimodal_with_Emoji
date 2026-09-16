"""Load the checked-in tweet/label CSVs without pandas.

The 2023 reader in ``data_utils.ReadOpen`` used ``encoding='utf-8'`` and
``errors='replace'``. This module does the same and adds split-level
statistics that the notebooks never printed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
SPLITS = ("train", "test", "subtest")


@dataclass(frozen=True)
class Split:
    name: str
    sentences: List[str]
    labels: List[int]
    path_sentences: Path
    path_labels: Path

    @property
    def n(self) -> int:
        return len(self.sentences)

    @property
    def label_counts(self) -> Dict[int, int]:
        return dict(Counter(self.labels))

    @property
    def n_positive(self) -> int:
        return int(sum(self.labels))

    @property
    def n_negative(self) -> int:
        return self.n - self.n_positive

    @property
    def positive_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return self.n_positive / self.n

    def pairs(self) -> Iterator[Tuple[str, int]]:
        return zip(self.sentences, self.labels)

    def head(self, k: int = 5) -> List[Tuple[str, int]]:
        return list(self.pairs())[:k]


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.rstrip("\n\r") for line in text.splitlines()]


def load_split(name: str, dataset_dir: Path | None = None) -> Split:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    sent_path = root / f"{name}_sentence.csv"
    label_path = root / f"{name}_label.csv"
    sentences = _read_lines(sent_path)
    raw_labels = [line.strip() for line in _read_lines(label_path) if line.strip() != ""]
    if len(sentences) != len(raw_labels):
        raise ValueError(
            f"{name}: {len(sentences)} sentences vs {len(raw_labels)} labels"
        )
    try:
        labels = [int(value) for value in raw_labels]
    except ValueError as exc:
        raise ValueError(f"{name}: labels must be integers") from exc
    if any(label not in (0, 1) for label in labels):
        bad = sorted({label for label in labels if label not in (0, 1)})
        raise ValueError(f"{name}: unexpected label values {bad}")
    return Split(
        name=name,
        sentences=sentences,
        labels=labels,
        path_sentences=sent_path,
        path_labels=label_path,
    )


def iter_splits(dataset_dir: Path | None = None) -> Iterator[Split]:
    for name in SPLITS:
        yield load_split(name, dataset_dir=dataset_dir)


def labels_are_blocked(labels: Sequence[int]) -> bool:
    """Return True if the label sequence is one run of 0s and one run of 1s."""
    if not labels:
        return True
    changes = sum(a != b for a, b in zip(labels, labels[1:]))
    return changes <= 1


@dataclass
class CueStats:
    n: int
    has_hashtag: int = 0
    has_mention: int = 0
    has_emoji: int = 0
    has_not_hashtag: int = 0
    has_sarcasm_token: int = 0
    emoji_in_negative: int = 0
    emoji_in_positive: int = 0
    word_lengths: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        n = max(self.n, 1)
        lengths = self.word_lengths or [0]
        ordered = sorted(lengths)
        return {
            "n": self.n,
            "hashtag_rate": self.has_hashtag / n,
            "mention_rate": self.has_mention / n,
            "emoji_rate": self.has_emoji / n,
            "not_hashtag_rate": self.has_not_hashtag / n,
            "sarcasm_token_rate": self.has_sarcasm_token / n,
            "emoji_in_negative": self.emoji_in_negative,
            "emoji_in_positive": self.emoji_in_positive,
            "words_min": min(lengths),
            "words_median": ordered[len(ordered) // 2],
            "words_max": max(lengths),
            "words_mean": sum(lengths) / max(len(lengths), 1),
        }


def _has_mention(text: str) -> bool:
    lowered = text.lower()
    return "<user>" in lowered or "@" in text


def _has_not_hashtag(text: str) -> bool:
    lowered = text.lower()
    return "#not" in lowered or "# not" in lowered


def _has_sarcasm_token(text: str) -> bool:
    return "sarcas" in text.lower()


# Broad but dependency-free emoji ranges. Good enough for split statistics.
_EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0x1F1E6, 0x1F1FF),
)


def is_emoji_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return any(start <= code <= end for start, end in _EMOJI_RANGES)


def tweet_has_emoji(text: str) -> bool:
    return any(is_emoji_char(char) for char in text)


def count_emoji(text: str) -> int:
    return sum(1 for char in text if is_emoji_char(char))


def cue_stats(split: Split) -> CueStats:
    stats = CueStats(n=split.n)
    for sentence, label in split.pairs():
        words = sentence.split()
        stats.word_lengths.append(len(words))
        if "#" in sentence:
            stats.has_hashtag += 1
        if _has_mention(sentence):
            stats.has_mention += 1
        has_emoji = tweet_has_emoji(sentence)
        if has_emoji:
            stats.has_emoji += 1
            if label == 0:
                stats.emoji_in_negative += 1
            else:
                stats.emoji_in_positive += 1
        if _has_not_hashtag(sentence):
            stats.has_not_hashtag += 1
        if _has_sarcasm_token(sentence):
            stats.has_sarcasm_token += 1
    return stats


def emoji_subset_indices(sentences: Sequence[str]) -> List[int]:
    return [index for index, sentence in enumerate(sentences) if tweet_has_emoji(sentence)]


def summarize_split(split: Split) -> Dict[str, object]:
    stats = cue_stats(split).as_dict()
    stats.update(
        {
            "name": split.name,
            "n_positive": split.n_positive,
            "n_negative": split.n_negative,
            "positive_rate": split.positive_rate,
            "labels_blocked": labels_are_blocked(split.labels),
            "first_labels": list(split.labels[:8]),
            "last_labels": list(split.labels[-8:]),
        }
    )
    return stats


def format_summary(summary: Dict[str, object]) -> str:
    lines = [
        f"split            {summary['name']}",
        f"n                {summary['n']}",
        f"literal / sarc   {summary['n_negative']} / {summary['n_positive']}",
        f"positive rate    {summary['positive_rate']:.4f}",
        f"labels blocked   {summary['labels_blocked']}",
        f"word tokens      min {summary['words_min']}  "
        f"median {summary['words_median']}  max {summary['words_max']}  "
        f"mean {summary['words_mean']:.2f}",
        f"hashtag rate     {summary['hashtag_rate']:.3f}",
        f"mention rate     {summary['mention_rate']:.3f}",
        f"emoji rate       {summary['emoji_rate']:.3f}  "
        f"(neg {summary['emoji_in_negative']}, pos {summary['emoji_in_positive']})",
        f"#not rate        {summary['not_hashtag_rate']:.3f}",
        f"sarcas* rate     {summary['sarcasm_token_rate']:.3f}",
        f"first labels     {summary['first_labels']}",
        f"last labels      {summary['last_labels']}",
    ]
    return "\n".join(lines)


def assert_subtest_is_test_emoji(test: Split, subtest: Split) -> None:
    """The course subtest is the emoji-bearing slice of the official test set."""
    indices = emoji_subset_indices(test.sentences)
    filtered_sentences = [test.sentences[i] for i in indices]
    filtered_labels = [test.labels[i] for i in indices]
    if filtered_sentences != subtest.sentences:
        raise AssertionError("subtest sentences are not the emoji subset of test")
    if filtered_labels != subtest.labels:
        raise AssertionError("subtest labels are not the emoji subset of test")


def batched(items: Sequence, size: int) -> Iterable[Sequence]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]
