from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from examples.lib.dataset_io import load_split
from examples.lib.lexical_features import FEATURE_NAMES, featurize, featurize_corpus
from examples.lib.logistic import LogisticBinary, binary_metrics

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_T = ROOT / "examples" / "fixtures" / "sample_tweets.csv"
FIXTURE_L = ROOT / "examples" / "fixtures" / "sample_labels.csv"


class FeatureTests(unittest.TestCase):
    def test_schema(self) -> None:
        vec = featurize("I loovee when people text back ... 😒 #sarcastictweet")
        self.assertEqual(vec.shape, (len(FEATURE_NAMES),))
        self.assertEqual(vec[0], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("has_sarcastic_tag")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("has_face_unamused")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("has_elongation")], 1.0)

    def test_not_tag_and_contrast(self) -> None:
        vec = featurize("I love getting home #not")
        self.assertEqual(vec[FEATURE_NAMES.index("has_not_tag")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("has_love")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("has_positive_word_plus_neg_tag")], 1.0)

    def test_corpus_stack(self) -> None:
        X = featurize_corpus(["a", "b #not"])
        self.assertEqual(X.shape, (2, len(FEATURE_NAMES)))


class LogisticTests(unittest.TestCase):
    def test_separable_cues(self) -> None:
        split = load_split(FIXTURE_T, FIXTURE_L)
        X = featurize_corpus(split.texts)
        y = np.asarray(split.labels)
        model = LogisticBinary(epochs=40, lr=0.15, batch_size=4, seed=1).fit(X, y)
        pred = model.predict(X)
        metrics = binary_metrics(y, pred)
        # 10 hand-picked rows with loud #not / #sarcasm cues should be learnable.
        self.assertGreaterEqual(metrics["accuracy"], 0.8)
        self.assertEqual(model.weights_.shape, (len(FEATURE_NAMES),))

    def test_metrics_known(self) -> None:
        y = np.asarray([1, 1, 0, 0])
        pred = np.asarray([1, 0, 0, 0])
        m = binary_metrics(y, pred)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
