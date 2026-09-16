from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from examples import (
    attention_walkthrough,
    embedding_matrix_walkthrough,
    inspect_dataset,
    lexical_baseline,
    multimodal_fusion,
    reported_results,
    tokenize_tweets,
)
from examples.logreg import binary_scores, fit_logreg, predict_label


class LogregTests(unittest.TestCase):
    def test_separable_two_features(self) -> None:
        rng = np.random.default_rng(1)
        x_pos = rng.normal(loc=2.0, size=(40, 2))
        x_neg = rng.normal(loc=-2.0, size=(40, 2))
        x = np.vstack([x_pos, x_neg])
        y = np.array([1] * 40 + [0] * 40)
        fit = fit_logreg(x, y, epochs=80, lr=0.2, l2=1e-4, seed=1)
        pred = predict_label(x, fit.weights, fit.bias)
        scores = binary_scores(y, pred)
        self.assertGreaterEqual(scores["accuracy"], 0.95)
        self.assertGreater(scores["f1"], 0.9)

    def test_scores_on_known_confusion(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        scores = binary_scores(y_true, y_pred)
        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fn"], 1)
        self.assertEqual(scores["tn"], 2)
        self.assertEqual(scores["fp"], 0)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["recall"], 0.5)


class ScriptSmokeTests(unittest.TestCase):
    def _run(self, func, args) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = func(args)
        self.assertEqual(code, 0, buf.getvalue())
        return buf.getvalue()

    def test_inspect_dataset_subtest(self) -> None:
        out = self._run(inspect_dataset.main, ["--split", "subtest", "--examples", "1"])
        self.assertIn("n                278", out)
        self.assertIn("check: subtest is the emoji-bearing subset of test", out)

    def test_reported_results(self) -> None:
        out = self._run(reported_results.main, ["--metric", "accuracy"])
        self.assertIn("Bi-LSTM + attention", out)
        self.assertIn("subtest", out.lower())

    def test_tokenize_tweets(self) -> None:
        out = self._run(
            tokenize_tweets.main,
            ["--split", "subtest", "--limit", "2", "--label", "sarc"],
        )
        self.assertIn("[sarc]", out)
        self.assertIn("#", out)

    def test_lexical_baseline_small(self) -> None:
        out = self._run(
            lexical_baseline.main,
            ["--max-train", "400", "--epochs", "40"],
        )
        self.assertIn("test   :", out)
        self.assertIn("subtest:", out)
        self.assertIn("has_not_hashtag", out)

    def test_attention_walkthrough(self) -> None:
        out = self._run(attention_walkthrough.main, [])
        self.assertIn("#not", out)
        masked = self._run(attention_walkthrough.main, ["--mask-last"])
        self.assertIn("#not", masked)

    def test_fusion_and_embedding(self) -> None:
        fused = self._run(multimodal_fusion.main, [])
        self.assertIn("fused shape (8,)", fused)
        emb = self._run(embedding_matrix_walkthrough.main, [])
        self.assertIn("word+emoji fallback", emb)
        self.assertIn("emoji fallbacks filled", emb)


if __name__ == "__main__":
    unittest.main()
