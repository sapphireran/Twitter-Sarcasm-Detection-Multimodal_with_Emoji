"""Load the three CSV splits and the integrity facts the docs rely on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ccs2lab.paths import SPLIT_FILES


@dataclass(frozen=True)
class Split:
    name: str
    texts: tuple[str, ...]
    labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_sarcastic(self) -> int:
        return sum(self.labels)

    @property
    def n_sincere(self) -> int:
        return len(self.labels) - self.n_sarcastic

    def items(self) -> Iterator[tuple[str, int]]:
        return zip(self.texts, self.labels, strict=True)


@dataclass(frozen=True)
class SplitBundle:
    train: Split
    test: Split
    subtest: Split

    def by_name(self, name: str) -> Split:
        try:
            return {"train": self.train, "test": self.test, "subtest": self.subtest}[name]
        except KeyError as exc:
            raise KeyError(f"unknown split {name!r}") from exc

    def all_splits(self) -> tuple[Split, Split, Split]:
        return self.train, self.test, self.subtest


def _read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def _read_labels(path: Path) -> list[int]:
    labels: list[int] = []
    for index, raw in enumerate(_read_lines(path), start=1):
        if raw == "":
            raise ValueError(f"{path}: empty label on line {index}")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: non-integer label {raw!r} on line {index}") from exc
        if value not in (0, 1):
            raise ValueError(f"{path}: label {value} on line {index} is not 0/1")
        labels.append(value)
    return labels


def load_split(name: str, *, sentence_path: Path | None = None, label_path: Path | None = None) -> Split:
    if name not in SPLIT_FILES and (sentence_path is None or label_path is None):
        raise KeyError(f"unknown split {name!r}")
    default_sent, default_lab = SPLIT_FILES.get(name, (None, None))
    sent_path = sentence_path or default_sent
    lab_path = label_path or default_lab
    if sent_path is None or lab_path is None:
        raise ValueError("sentence_path and label_path are required for custom splits")
    texts = tuple(_read_lines(sent_path))
    labels = tuple(_read_labels(lab_path))
    if len(texts) != len(labels):
        raise ValueError(
            f"{name}: {len(texts)} sentences vs {len(labels)} labels"
        )
    return Split(name=name, texts=texts, labels=labels)


def load_bundle() -> SplitBundle:
    return SplitBundle(
        train=load_split("train"),
        test=load_split("test"),
        subtest=load_split("subtest"),
    )


@dataclass(frozen=True)
class IntegrityReport:
    aligned: bool
    train_test_overlap: int
    test_subtest_overlap: int
    train_subtest_overlap: int
    subtest_subset_of_test: bool
    expected_sizes: dict[str, int]


def integrity(bundle: SplitBundle) -> IntegrityReport:
    train_set = set(bundle.train.texts)
    test_set = set(bundle.test.texts)
    sub_set = set(bundle.subtest.texts)
    return IntegrityReport(
        aligned=True,
        train_test_overlap=len(train_set & test_set),
        test_subtest_overlap=len(test_set & sub_set),
        train_subtest_overlap=len(train_set & sub_set),
        subtest_subset_of_test=sub_set <= test_set,
        expected_sizes={
            "train": 39780,
            "test": 2000,
            "subtest": 278,
        },
    )
