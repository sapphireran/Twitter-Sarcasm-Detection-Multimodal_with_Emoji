"""Guards on the shipped CSV pairs and the copied 2023 metric table."""

from __future__ import annotations

import csv

import pytest

from examples.lib.io import SPLITS, load_split, repo_root, split_paths

EXPECTED_ROWS = {"train": 39780, "test": 2000, "subtest": 278}


def test_sentence_and_label_files_line_up() -> None:
    root = repo_root()
    for name in SPLITS:
        sentence_path, label_path = split_paths(name, root)
        assert sentence_path.is_file(), sentence_path
        assert label_path.is_file(), label_path
        sentences = sentence_path.read_text(encoding="utf-8", errors="replace").splitlines()
        labels = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        assert len(sentences) == len(labels) == EXPECTED_ROWS[name]


def test_labels_are_binary_and_both_classes_present() -> None:
    for name in SPLITS:
        rows = load_split(name)
        labels = {label for _, label in rows}
        assert labels == {0, 1}
        assert all(isinstance(text, str) and text for text, _ in rows[:20])


def test_subtest_is_emoji_heavy() -> None:
    from examples.lib.tokenize import find_emoji

    rows = load_split("subtest")
    with_emoji = sum(1 for text, _ in rows if find_emoji(text))
    assert with_emoji / len(rows) > 0.95


def test_reported_metrics_csv_matches_notebook_highlights() -> None:
    path = repo_root() / "docs" / "reported_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    by_key = {(row["split"], row["model"], row["modality"]): row for row in rows}
    multi = by_key[("test", "bilstm_attention", "multi")]
    assert multi["accuracy"] == "0.8735"
    assert float(multi["f1"]) == pytest.approx(0.8685714285714285)
    sub = by_key[("subtest", "bilstm_attention", "multi")]
    assert float(sub["accuracy"]) == pytest.approx(0.8920863309352518)
