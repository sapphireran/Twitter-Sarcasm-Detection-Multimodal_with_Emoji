from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import examples.attention_demo as attention_demo
import examples.embedding_average as embedding_average
import examples.emoji_signals as emoji_signals
import examples.inspect_dataset as inspect_dataset
import examples.lexical_baseline as lexical_baseline
import examples.tokenize_tweets as tokenize_tweets

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "examples" / "fixtures"


class CliSmokeTests(unittest.TestCase):
    def test_attention_demo(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = attention_demo.main([])
        self.assertEqual(rc, 0)
        self.assertIn("alpha", buf.getvalue())

    def test_embedding_average(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = embedding_average.main([])
        self.assertEqual(rc, 0)
        self.assertIn("dim(fused)", buf.getvalue())

    def test_tokenize(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tokenize_tweets.main([])
        self.assertEqual(rc, 0)
        self.assertIn("#sarcastictweet", buf.getvalue())

    def test_inspect_fixtures_via_real_root_optional(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = inspect_dataset.main(["--root", str(ROOT / "dataset")])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("39780", out)
        self.assertIn("278 / 278", out)

    def test_emoji_signals_min_support(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = emoji_signals.main(
                ["--root", str(ROOT / "dataset"), "--min-support", "80", "--top-pairs", "3"]
            )
        self.assertEqual(rc, 0)
        self.assertIn("PMI", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
