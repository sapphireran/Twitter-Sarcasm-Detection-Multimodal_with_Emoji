"""Tiny extra tests that execute the example scripts' imports."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExampleImportTests(unittest.TestCase):
    def test_reported_json_is_valid(self) -> None:
        # Load the same way examples do.
        sys.path.insert(0, str(ROOT / "examples"))
        try:
            import _path  # noqa: F401
        finally:
            if str(ROOT / "examples") in sys.path:
                sys.path.remove(str(ROOT / "examples"))
        from sarcasm_toolkit.results import load_reported

        table = load_reported()
        self.assertEqual(table["splits"]["test"]["n"], 2000)
        self.assertIn("bilstm_attention", table["models"])

    def test_toolkit_main_module(self) -> None:
        module = importlib.import_module("sarcasm_toolkit.__main__")
        self.assertTrue(callable(module.main))


class AttentionExampleChecksum(unittest.TestCase):
    def test_demo_weights_sum_to_one(self) -> None:
        from sarcasm_toolkit.attention import attention_report
        from sarcasm_toolkit.embeddings import embed_tokens
        from sarcasm_toolkit.tokenize import tokenize_tweet

        w = [0.05, 0.05, 1.40, 1.10, 0.15, 0.05, 0.05, 0.05]
        tokens = tokenize_tweet("I just love having grungy ass hair 😑 #not")
        total = sum(
            row["weight"]
            for row in attention_report(embed_tokens(tokens), w, tokens=tokens)
        )
        self.assertAlmostEqual(total, 1.0, places=5)
