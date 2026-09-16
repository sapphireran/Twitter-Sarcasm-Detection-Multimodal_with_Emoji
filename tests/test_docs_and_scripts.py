from pathlib import Path

from examples.lib.io import repo_root

DOC_FILES = (
    "README.md",
    "docs/README.md",
    "docs/overview.md",
    "docs/dataset.md",
    "docs/architecture.md",
    "docs/attention.md",
    "docs/baselines-and-results.md",
    "docs/reproduction.md",
    "docs/examples.md",
    "docs/reported_metrics.csv",
    "examples/README.md",
)

SCRIPTS = (
    "examples/01_dataset_preview.py",
    "examples/02_lexical_cues.py",
    "examples/03_emoji_vectors.py",
    "examples/04_attention_walkthrough.py",
    "examples/05_tfidf_baseline.py",
    "examples/run_all.sh",
)


def test_writeup_files_exist_and_are_nontrivial() -> None:
    root = repo_root()
    for relative in DOC_FILES:
        path = root / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 200, relative


def test_scripts_are_executable_python() -> None:
    root = repo_root()
    for relative in SCRIPTS:
        path = root / relative
        assert path.is_file(), relative
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            assert text.startswith("#!/usr/bin/env python3")
            assert "if __name__" in text
        else:
            assert "set -euo pipefail" in text


def test_readme_points_at_docs_and_examples() -> None:
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    assert "docs/baselines-and-results.md" in readme
    assert "examples/01_dataset_preview.py" in readme
    assert "Personal 2023" in readme
