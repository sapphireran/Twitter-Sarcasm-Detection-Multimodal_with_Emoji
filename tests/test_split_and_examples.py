"""CSV alignment plus smoke-running the example entry points."""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import read_sentence_label_pair


def _load(name: str, filename: str):
    path = ROOT / "examples" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitFileTests(unittest.TestCase):
    def test_each_official_pair_aligns(self) -> None:
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for split, n in expected.items():
            docs, labels = read_sentence_label_pair(
                ROOT / "dataset" / f"{split}_sentence.csv",
                ROOT / "dataset" / f"{split}_label.csv",
            )
            self.assertEqual(len(docs), n)
            self.assertEqual(len(labels), n)
            self.assertTrue(set(labels.tolist()).issubset({0, 1}))


class ExampleMainTests(unittest.TestCase):
    def _run(self, filename: str) -> str:
        module = _load(f"ex_{filename}", filename)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = module.main()
        self.assertEqual(code, 0, msg=buf.getvalue())
        return buf.getvalue()

    def test_overview_mentions_all_splits(self) -> None:
        out = self._run("01_dataset_overview.py")
        for split in ("train", "test", "subtest"):
            self.assertIn(split, out)

    def test_tokenize_reports_fixture_size(self) -> None:
        out = self._run("02_tokenize_tweets.py")
        self.assertIn("fixture tweets: 12", out)

    def test_cues_prints_rule(self) -> None:
        out = self._run("03_sarcasm_cues.py")
        self.assertIn("cue-as-predictor", out)
        self.assertIn("FP=0", out)  # test split has no cue false positives

    def test_tiny_embedding_shapes(self) -> None:
        out = self._run("04_tiny_embedding_pipeline.py")
        self.assertIn("word view   (12, 16)", out)
        self.assertIn("multi view  (12, 32)", out)

    def test_attention_example_passes_its_checks(self) -> None:
        out = self._run("05_attention_math.py")
        self.assertIn("checks passed", out)

    def test_toy_baseline_scores_test(self) -> None:
        out = self._run("06_toy_baseline.py")
        self.assertIn("test", out)
        self.assertIn("cue⇒sarcastic", out)

    def test_split_consistency_script(self) -> None:
        out = self._run("07_split_consistency.py")
        self.assertIn("all split consistency checks passed", out)

    def test_results_table(self) -> None:
        out = self._run("08_format_results_table.py")
        self.assertIn("BiLSTM + attention", out)
        self.assertIn("0.8735", out)


if __name__ == "__main__":
    unittest.main()
