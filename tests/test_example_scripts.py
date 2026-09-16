from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = (
    "examples/inspect_dataset.py",
    "examples/tokenize_tweets.py",
    "examples/toy_embedding_fusion.py",
    "examples/attention_walkthrough.py",
    "examples/lexical_sarcasm_baseline.py",
)


class ExampleScriptTests(unittest.TestCase):
    def test_scripts_exit_zero(self) -> None:
        for rel in SCRIPTS:
            extra = []
            if rel.endswith("lexical_sarcasm_baseline.py"):
                extra = ["--limit-train", "4000", "--epochs", "40"]
            proc = subprocess.run(
                [sys.executable, str(ROOT / rel), *extra],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"{rel} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            self.assertTrue(proc.stdout.strip(), msg=f"{rel} printed nothing")

    def test_attention_mask_flag(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples/attention_walkthrough.py"), "--mask-last"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("masked", proc.stdout.lower())

    def test_inspect_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples/inspect_dataset.py"), "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"train"', proc.stdout)
        self.assertIn("39780", proc.stdout)
        self.assertIn("hashtag_rule", proc.stdout)


if __name__ == "__main__":
    unittest.main()
