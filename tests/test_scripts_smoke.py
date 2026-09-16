"""Sanity checks that the example scripts exit 0 and print expected headers."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(ROOT / "examples" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestScriptsSmoke(unittest.TestCase):
    def test_explore(self):
        proc = run("01_explore_dataset.py")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("subtest", proc.stdout)
        self.assertIn("39780", proc.stdout)

    def test_explore_json(self):
        proc = run("01_explore_dataset.py", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"n": 39780', proc.stdout)

    def test_lexical_cues(self):
        proc = run("03_lexical_cues.py")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("rule acc", proc.stdout)

    def test_toy_attention(self):
        proc = run("05_toy_attention.py", "--steps", "5", "--peak", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("argmax(alpha) = 2", proc.stdout)
        self.assertIn("sum(alpha) = 1", proc.stdout)

    def test_emoji_cooccurrence(self):
        proc = run("02_emoji_cooccurrence.py", "--split", "subtest", "--top", "5")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PMI", proc.stdout)

    def test_inspect_emoji2vec(self):
        proc = run("04_inspect_emoji2vec.py", "--query", "❤", "😒")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("1661", proc.stdout)
        self.assertIn("resolved via U+FE0F", proc.stdout)

    def test_bow_baseline(self):
        proc = run("06_bow_baseline.py", "--max-features", "500", "--min-count", "5")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("stripped", proc.stdout)
        self.assertIn("emoji_flag", proc.stdout)

    def test_average_pooling(self):
        proc = run("07_average_pooling.py", "--n", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("WE dim", proc.stdout)

    def test_results_table(self):
        proc = run("08_results_table.py", "--metric", "accuracy")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("bilstm_attention", proc.stdout)
        self.assertIn("87.35", proc.stdout)


if __name__ == "__main__":
    unittest.main()
