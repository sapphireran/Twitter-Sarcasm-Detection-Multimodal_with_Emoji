import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_example(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / args[0]), *args[1:]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


class ExampleScriptTests(unittest.TestCase):
    def test_inspect_dataset_json(self) -> None:
        proc = run_example("examples/inspect_dataset.py", "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["test"]["n"], 2000)
        self.assertEqual(payload["subtest"]["n"], 278)

    def test_heuristic_json_matches_published_test_precision(self) -> None:
        proc = run_example("examples/heuristic_baseline.py", "--json")
        rows = {row["split"]: row for row in json.loads(proc.stdout)}
        self.assertEqual(rows["test"]["precision"], 1.0)
        self.assertGreater(rows["subtest"]["f1"], 0.85)

    def test_predict_cli_on_sample_file(self) -> None:
        proc = run_example(
            "examples/predict_cli.py",
            "--json",
            "--file",
            "examples/sample_tweets.txt",
        )
        payload = json.loads(proc.stdout)
        labels = [row["label"] for row in payload]
        self.assertIn("sarcastic", labels)
        self.assertIn("non-sarcastic", labels)

    def test_attention_walkthrough_mentions_not(self) -> None:
        proc = run_example("examples/attention_walkthrough.py")
        self.assertIn("#not", proc.stdout)
        self.assertIn("context vector", proc.stdout)
