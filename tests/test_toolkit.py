"""Unit tests for the stdlib sarcasm toolkit used by docs/examples."""

from __future__ import annotations

import math
import unittest

from sarcasm_toolkit.attention import attention_pool, attention_weights, softmax
from sarcasm_toolkit.baseline import CueLogistic, LexiconBaseline
from sarcasm_toolkit.cues import CUE_HASHTAGS, extract_cue_features, feature_names
from sarcasm_toolkit.dataset import Split, load_split, sample_split, summarize_split
from sarcasm_toolkit.embeddings import average_pool, embed_token, embed_tokens
from sarcasm_toolkit.metrics import binary_metrics, confusion, majority_baseline
from sarcasm_toolkit.results import iter_reported_rows, load_reported
from sarcasm_toolkit.tokenize import (
    emoji_tokens,
    has_elongation,
    hashtags,
    tokenize_tweet,
)


class TokenizeTests(unittest.TestCase):
    def test_hashtag_and_emoji_stay_whole(self) -> None:
        tokens = tokenize_tweet("I just love Mondays #not 😒")
        self.assertEqual(tokens, ["i", "just", "love", "mondays", "#not", "😒"])
        self.assertEqual(hashtags(tokens), ["#not"])
        self.assertEqual(emoji_tokens(tokens), ["😒"])

    def test_user_placeholder_and_url(self) -> None:
        tokens = tokenize_tweet("<user> see https://example.com/a #YeahRight")
        self.assertIn("<user>", tokens)
        self.assertIn("https://example.com/a", tokens)
        self.assertIn("#yeahright", tokens)

    def test_elongation(self) -> None:
        self.assertTrue(has_elongation("loovee"))
        self.assertTrue(has_elongation("yayyy"))
        self.assertTrue(has_elongation("sooo happy"))
        self.assertFalse(has_elongation("love"))
        self.assertFalse(has_elongation("looking good"))

    def test_lone_hash_is_not_a_hashtag(self) -> None:
        tokens = tokenize_tweet("wait # what #not")
        self.assertEqual(hashtags(tokens), ["#not"])

    def test_empty(self) -> None:
        self.assertEqual(tokenize_tweet(""), [])


class CueTests(unittest.TestCase):
    def test_feature_vector_width(self) -> None:
        names = feature_names()
        feats = extract_cue_features("I just love having grungy ass hair 😑 #not")
        self.assertEqual(len(names), 17)
        self.assertEqual(len(feats.values), 17)
        named = dict(zip(names, feats.values))
        self.assertEqual(named["has_not_hashtag"], 1.0)
        self.assertEqual(named["any_cue_hashtag"], 1.0)
        self.assertEqual(named["has_emoji"], 1.0)
        self.assertEqual(named["positive_valence"], 1.0)

    def test_cue_set_contains_core_tags(self) -> None:
        self.assertTrue({"#not", "#sarcasm", "#sarcastictweet"} <= CUE_HASHTAGS)


class AttentionTests(unittest.TestCase):
    def test_softmax_sums_to_one(self) -> None:
        weights = softmax([0.2, 0.5, -1.0])
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_mask_zeros_out_a_timestep(self) -> None:
        weights = softmax([2.0, 2.0, 2.0], mask=[1.0, 0.0, 1.0])
        self.assertAlmostEqual(weights[1], 0.0, places=9)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_pool_matches_weighted_sum(self) -> None:
        sequence = [[1.0, 0.0], [0.0, 1.0]]
        w = [1.0, 0.0]
        alphas = attention_weights(sequence, w)
        pooled = attention_pool([sequence], w)[0]
        expected = [
            alphas[0] * sequence[0][0] + alphas[1] * sequence[1][0],
            alphas[0] * sequence[0][1] + alphas[1] * sequence[1][1],
        ]
        self.assertEqual(len(alphas), 2)
        self.assertAlmostEqual(sum(alphas), 1.0, places=5)
        self.assertAlmostEqual(pooled[0], expected[0], places=9)
        self.assertAlmostEqual(pooled[1], expected[1], places=9)

    def test_known_energies(self) -> None:
        sequence = [[1.0, 0.0], [0.0, 1.0]]
        alphas = attention_weights(sequence, [1.0, 0.0])
        # e = tanh([1, 0]) → first timestep should dominate.
        self.assertGreater(alphas[0], alphas[1])


class MetricsTests(unittest.TestCase):
    def test_perfect_and_none(self) -> None:
        perfect = binary_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(perfect.accuracy, 1.0)
        self.assertEqual(perfect.f1, 1.0)
        none = binary_metrics([1, 1, 0, 0], [0, 0, 0, 0])
        self.assertEqual(none.true_positive, 0)
        self.assertEqual(none.recall, 0.0)

    def test_confusion_order(self) -> None:
        tp, fp, tn, fn = confusion([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((tp, fp, tn, fn), (1, 1, 1, 1))

    def test_majority(self) -> None:
        self.assertEqual(majority_baseline([0, 0, 1]), [0, 0, 0])
        self.assertEqual(majority_baseline([1, 1, 0]), [1, 1, 1])


class EmbeddingTests(unittest.TestCase):
    def test_seed_token_is_not_hashed(self) -> None:
        vec = embed_token("#not")
        self.assertEqual(vec[2], 1.0)

    def test_average_pool_empty_is_zeros(self) -> None:
        self.assertEqual(average_pool([]), [0.0] * 8)

    def test_embed_tokens_width(self) -> None:
        vectors = embed_tokens(["love", "#not", "😒"])
        self.assertEqual(len(vectors), 3)
        self.assertTrue(all(len(row) == 8 for row in vectors))


class BaselineTests(unittest.TestCase):
    def test_lexicon_perfect_precision_on_test(self) -> None:
        split = load_split("test")
        metrics = binary_metrics(split.labels, LexiconBaseline().predict(split.texts))
        self.assertEqual(metrics.false_positive, 0)
        self.assertGreater(metrics.true_positive, 500)
        self.assertAlmostEqual(metrics.accuracy, 0.807, places=3)
        model = LexiconBaseline()
        self.assertEqual(model.predict_one("I just love Mondays #not"), 1)
        self.assertEqual(model.predict_one("i just imagined you dancing like this"), 0)
        self.assertEqual(model.predict_one("so sarcastic without a tag"), 0)

    def test_logistic_fits_separable_cues(self) -> None:
        texts = [
            "hate this #not",
            "great day #sarcasm",
            "love when it rains #sarcastictweet",
            "see you tomorrow",
            "had lunch with friends",
            "the train is on time",
        ]
        labels = [1, 1, 1, 0, 0, 0]
        model = CueLogistic(epochs=80, learning_rate=0.5, l2=0.0)
        model.fit(texts, labels)
        preds = model.predict(texts)
        self.assertEqual(preds, labels)


class DatasetTests(unittest.TestCase):
    def test_split_sizes(self) -> None:
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, n in expected.items():
            split = load_split(name)
            self.assertEqual(len(split), n)
            self.assertEqual(len(split.texts), len(split.labels))
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_subtest_is_emoji_heavy(self) -> None:
        split = load_split("subtest")
        with_emoji = 0
        for example in split:
            if emoji_tokens(tokenize_tweet(example.text)):
                with_emoji += 1
        self.assertGreaterEqual(with_emoji / len(split), 0.95)

    def test_sample_split_is_mixed(self) -> None:
        train = load_split("train")
        # Prefix of train is literal-heavy; sampling must not use that prefix.
        prefix = Split(
            name="prefix",
            texts=train.texts[:3000],
            labels=train.labels[:3000],
        )
        sampled = sample_split(train, 3000, seed=2023)
        self.assertGreater(sampled.n_sarcastic, 500)
        self.assertLess(prefix.n_sarcastic, sampled.n_sarcastic)
        self.assertEqual(len(sampled), 3000)

    def test_summarize_keys(self) -> None:
        summary = summarize_split(load_split("test"))
        self.assertEqual(summary["n"], 2000)
        self.assertEqual(summary["sarcastic"], 1000)
        self.assertIn("token_len", summary)


class ReportedTests(unittest.TestCase):
    def test_json_matches_notebook_bilstm(self) -> None:
        table = load_reported()
        multi = table["models"]["bilstm_attention"]["test"]["multi"]
        self.assertAlmostEqual(multi["acc"], 0.8735, places=4)
        rows = iter_reported_rows()
        self.assertEqual(len(rows), 20)  # 5 models × 2 splits × 2 modes


if __name__ == "__main__":
    unittest.main()
