"""Unit tests for the personal example library."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.attention import attention_entropy, make_demo_sequence, raffel_attention, softmax
from sarcasm_lab.cue_model import RuleCueClassifier
from sarcasm_lab.cues import extract_cues
from sarcasm_lab.io import load_all_splits, load_split
from sarcasm_lab.metrics import binary_metrics
from sarcasm_lab.naive_bayes import MultinomialNB
from sarcasm_lab.paths import DATASET_DIR
from sarcasm_lab.sgd_logreg import SGDLogisticRegression
from sarcasm_lab.tokenize import is_emoji_token, tokenize_tweet
from sarcasm_lab.vectorize import CountVectorizer


class DatasetTests(unittest.TestCase):
    def test_committed_row_counts(self) -> None:
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, n in expected.items():
            split = load_split(name, tokenize=False)
            self.assertEqual(len(split), n, name)
            self.assertEqual(len(split.texts), len(split.labels))

    def test_test_set_is_balanced(self) -> None:
        split = load_split("test", tokenize=False)
        counts = split.label_counts()
        self.assertEqual(counts[0], 1000)
        self.assertEqual(counts[1], 1000)

    def test_dataset_dir_exists(self) -> None:
        self.assertTrue(DATASET_DIR.is_dir())
        self.assertTrue((DATASET_DIR / "train_sentence.csv").is_file())

    def test_all_splits_round_trip(self) -> None:
        splits = load_all_splits(tokenize=True)
        self.assertEqual(set(splits), {"train", "test", "subtest"})
        self.assertGreater(len(splits["train"].tokens[0]), 0)


class TokenizeTests(unittest.TestCase):
    def test_hashtag_and_emoji(self) -> None:
        tokens = tokenize_tweet("I love watching golf ! ⛳ #not")
        self.assertIn("#not", tokens)
        self.assertTrue(any(is_emoji_token(tok) for tok in tokens))
        self.assertIn("love", tokens)

    def test_user_and_url(self) -> None:
        tokens = tokenize_tweet("<user> see https://example.com/a tonight")
        self.assertIn("<user>", tokens)
        self.assertIn("<url>", tokens)

    def test_comma_becomes_space(self) -> None:
        tokens = tokenize_tweet("one,two")
        self.assertEqual(tokens, ["one", "two"])

    def test_ellipsis(self) -> None:
        tokens = tokenize_tweet("text back ... please")
        self.assertIn("...", tokens)


class CueTests(unittest.TestCase):
    def test_sarcasm_hashtag_flag(self) -> None:
        text = "I love walking to school #not"
        cues = extract_cues(text, tokenize_tweet(text))
        self.assertTrue(cues.has_sarcasm_hashtag)
        self.assertTrue(cues.has_positive_stem)
        self.assertIn("#not", cues.sarcasm_hashtags)

    def test_rule_classifier_on_subtest_style_line(self) -> None:
        text = "I loovee when people text back ... 😒 #sarcastictweet"
        cues = extract_cues(text, tokenize_tweet(text))
        self.assertEqual(RuleCueClassifier().predict_one(cues), 1)


class MetricsTests(unittest.TestCase):
    def test_perfect_scores(self) -> None:
        m = binary_metrics([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(m.accuracy, 1.0)
        self.assertEqual(m.f1, 1.0)

    def test_all_negative_predictions(self) -> None:
        m = binary_metrics([1, 1, 0, 0], [0, 0, 0, 0])
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.true_negative, 2)
        self.assertEqual(m.false_negative, 2)


class VectorizerAndModelsTests(unittest.TestCase):
    def test_nb_separates_toy_topic(self) -> None:
        docs = [
            ["love", "this", "song"],
            ["love", "this", "album"],
            ["hate", "#not", "this"],
            ["hate", "#not", "today"],
        ]
        y = [0, 0, 1, 1]
        vec = CountVectorizer(min_df=1, max_features=20)
        X = vec.fit_transform(docs)
        model = MultinomialNB().fit(X, y, vec.n_features)
        pred = model.predict(vec.transform([["hate", "#not"], ["love", "song"]]))
        self.assertEqual(pred, [1, 0])

    def test_logreg_learns_hashtag(self) -> None:
        docs = [["fun", "day"]] * 8 + [["fun", "day", "#not"]] * 8
        y = [0] * 8 + [1] * 8
        vec = CountVectorizer(min_df=1, max_features=20)
        X = vec.fit_transform(docs)
        model = SGDLogisticRegression(epochs=12, lr=0.2, l2=0.0, seed=1)
        model.fit(X, y, vec.n_features)
        pred = model.predict(vec.transform([["fun", "day"], ["fun", "day", "#not"]]))
        self.assertEqual(pred, [0, 1])


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one_and_respect_mask(self) -> None:
        demo = make_demo_sequence()
        attn = demo["attention"][0]
        self.assertTrue(math.isclose(float(attn.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5))
        self.assertLess(float(attn[4]), 1e-8)
        self.assertLess(float(attn[5]), 1e-8)
        self.assertGreater(float(attn[-1] + attn[-2]), float(attn[0] + attn[1]))

    def test_softmax_flat_logits(self) -> None:
        weights = softmax(__import__("numpy").array([[0.0, 0.0, 0.0]]))
        self.assertTrue(all(math.isclose(float(w), 1 / 3, rel_tol=1e-6) for w in weights[0]))

    def test_context_shape(self) -> None:
        demo = make_demo_sequence()
        context, attn = raffel_attention(demo["x"], demo["weight"], demo["bias"], demo["mask"])
        self.assertEqual(context.shape, (1, 6))
        self.assertEqual(attn.shape, (1, 8))
        self.assertGreater(float(attention_entropy(attn)[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
