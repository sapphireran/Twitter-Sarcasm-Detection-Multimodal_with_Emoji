"""Smoke-import the example scripts as modules (they are executable files)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load(name: str):
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ScriptImportTests(unittest.TestCase):
    def test_scripts_define_main(self) -> None:
        for name in (
            "dataset_overview",
            "cue_analysis",
            "tokenize_demo",
            "attention_walkthrough",
            "lexical_baseline",
        ):
            module = _load(name)
            self.assertTrue(callable(getattr(module, "main", None)), name)
