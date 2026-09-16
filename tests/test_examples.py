"""Unit tests for the personal example library.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from examples.attention import attention_forward, attention_scores, masked_softmax
from examples.classify import accuracy, f1_score, fit_logreg, predict_label
from examples.dataset_io import (
    SPLIT_FILES,
    iter_split,
    read_labels,
    read_open_replica,
    read_sentence_strings,
)
from examples.dataset_overview import summarize_split
from examples.embeddings import (
    average_corpus,
    average_vector_per_sequence,
    build_embedding_matrix,
    multimodal_concat,
)
from examples.emoji import extract_emojis, has_emoji, tokens_that_are_emoji
from examples.fixtures.tiny_tables import DIM, EMOJI_TABLE, WORD_TABLE
from examples.tokenize import join_comma_split_line, tokenize_tweet

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "fixtures"


class TokenizeTests(unittest.TestCase):
    def test_keeps_hashtags_mentions_and_emoji(self) -> None:
        tokens = tokenize_tweet("Don't you love it #not 😒 @user")
        self.assertIn("#not", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("@user", tokens)
        self.assertIn("don't", tokens)

    def test_keeps_anonymized_user_placeholder(self) -> None:
        tokens = tokenize_tweet("<user> Rest in peace")
        self.assertIn("<user>", tokens)
        self.assertNotIn("<", tokens)

    def test_lowercases_like_readopen(self) -> None:
        tokens = tokenize_tweet("#SarcasticTweet I Know")
        self.assertEqual(tokens[:2], ["#sarcastictweet", "i"])

    def test_url_stays_one_token(self) -> None:
        tokens = tokenize_tweet("see https://example.com/a now")
        self.assertIn("https://example.com/a", tokens)

    def test_comma_rebuild_drops_quoted_comma(self) -> None:
        raw = '"So many useless classes , great to be student"'
        rebuilt = join_comma_split_line(raw)
        self.assertNotIn(",", rebuilt)
        self.assertTrue(rebuilt.startswith('"'))
        self.assertIn("great to be student", rebuilt)

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize_tweet(""), [])


class EmojiTests(unittest.TestCase):
    def test_extract_preserves_order(self) -> None:
        self.assertEqual(extract_emojis("hi 😒 then 😭"), ["😒", "😭"])

    def test_has_emoji_false_on_plain_text(self) -> None:
        self.assertFalse(has_emoji("plain text #not"))
        self.assertTrue(has_emoji("plain text 😅"))

    def test_token_filter(self) -> None:
        tokens = tokenize_tweet("love 😄 #not")
        self.assertEqual(tokens_that_are_emoji(tokens), ["😄"])


class DatasetIoTests(unittest.TestCase):
    def test_fixture_alignment(self) -> None:
        texts = read_sentence_strings(FIXTURES / "tiny_tweets.csv")
        labels = read_labels(FIXTURES / "tiny_labels.csv")
        self.assertEqual(len(texts), 16)
        self.assertEqual(len(labels), 16)
        self.assertEqual(labels.count(1), 8)
        self.assertEqual(labels.count(0), 8)

    def test_read_open_replica_tokenizes_fixture(self) -> None:
        docs, labels, count = read_open_replica(
            FIXTURES / "tiny_tweets.csv",
            FIXTURES / "tiny_labels.csv",
        )
        self.assertEqual(count, 16)
        self.assertEqual(len(docs), 16)
        self.assertEqual(labels[0], 1)
        self.assertIn("#sarcastictweet", docs[0])

    def test_project_splits_line_up(self) -> None:
        for name, (sentence_path, label_path) in SPLIT_FILES.items():
            texts, labels = iter_split(name)
            self.assertEqual(len(texts), len(labels), name)
            self.assertTrue(sentence_path.is_file(), sentence_path)
            self.assertTrue(label_path.is_file(), label_path)

    def test_csv_reader_keeps_quoted_comma(self) -> None:
        test_texts = read_sentence_strings(ROOT / "dataset" / "test_sentence.csv")
        quoted = next(text for text in test_texts if "useless classes" in text)
        self.assertIn(",", quoted)


class DatasetOverviewTests(unittest.TestCase):
    def test_known_split_sizes(self) -> None:
        train = summarize_split("train")
        test = summarize_split("test")
        subtest = summarize_split("subtest")
        self.assertEqual(train["n"], 39780)
        self.assertEqual(train["n_non_sarcastic"], 21292)
        self.assertEqual(train["n_sarcastic"], 18488)
        self.assertEqual(test["n"], 2000)
        self.assertEqual(test["n_sarcastic"], 1000)
        self.assertEqual(subtest["n"], 278)
        self.assertEqual(train["n_with_emoji"], 5479)
        self.assertEqual(test["n_with_emoji"], 278)
        self.assertEqual(subtest["n_with_emoji"], 278)
        self.assertAlmostEqual(test["emoji_rate"], subtest["n"] / test["n"], places=3)

    def test_subtest_is_emoji_slice_of_test(self) -> None:
        test_texts, _ = iter_split("test")
        sub_texts, _ = iter_split("subtest")
        test_set = set(test_texts)
        missing = [text for text in sub_texts if text not in test_set]
        self.assertEqual(missing, [])


class EmbeddingTests(unittest.TestCase):
    def test_mean_skips_oov_and_zeros_when_empty(self) -> None:
        table = {"love": np.array([2.0, 0.0]), "hate": np.array([0.0, 2.0])}
        mixed = average_vector_per_sequence(["love", "unknown", "hate"], table)
        np.testing.assert_allclose(mixed, np.array([1.0, 1.0]))
        empty = average_vector_per_sequence(["zzzz"], table, dim=2)
        np.testing.assert_allclose(empty, np.zeros(2))

    def test_multimodal_concat(self) -> None:
        docs = [tokenize_tweet("I love this 😒 #not")]
        word = average_corpus(docs, WORD_TABLE, dim=DIM)
        emoji = average_corpus(docs, EMOJI_TABLE, dim=DIM)
        both = multimodal_concat(word, emoji)
        self.assertEqual(word.shape, (1, DIM))
        self.assertEqual(both.shape, (1, DIM * 2))
        self.assertGreater(both[0, 2], 0.0)  # sarcasm-tag axis
        self.assertGreater(both[0, DIM + 6], 0.0)  # deadpan emoji axis

    def test_embedding_matrix_emoji_fallback(self) -> None:
        word_index = {"love": 1, "😒": 2}
        matrix = build_embedding_matrix(
            word_index,
            vocab_size=4,
            word_table=WORD_TABLE,
            emoji_table=EMOJI_TABLE,
            dim=DIM,
            use_emoji_fallback=True,
        )
        np.testing.assert_allclose(matrix[1], WORD_TABLE["love"])
        np.testing.assert_allclose(matrix[2], EMOJI_TABLE["😒"])
        off = build_embedding_matrix(
            word_index,
            vocab_size=4,
            word_table=WORD_TABLE,
            emoji_table=EMOJI_TABLE,
            dim=DIM,
            use_emoji_fallback=False,
        )
        np.testing.assert_allclose(off[2], np.zeros(DIM))

    def test_concat_rejects_mismatched_batches(self) -> None:
        with self.assertRaises(ValueError):
            multimodal_concat(np.zeros((2, 3)), np.zeros((1, 3)))


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6, 5))
        weight = rng.normal(size=(5,))
        context, attn = attention_forward(x, weight)
        np.testing.assert_allclose(attn.sum(axis=1), np.ones(4), atol=1e-6)
        self.assertEqual(context.shape, (4, 5))

    def test_mask_zeros_out_steps(self) -> None:
        x = np.zeros((1, 3, 2))
        x[0, 0] = [1.0, 0.0]
        x[0, 1] = [0.0, 1.0]
        x[0, 2] = [5.0, 5.0]
        weight = np.array([1.0, 1.0])
        mask = np.array([[1.0, 1.0, 0.0]])
        _, attn = attention_forward(x, weight, mask=mask)
        self.assertAlmostEqual(attn[0, 2], 0.0, places=6)
        np.testing.assert_allclose(attn.sum(axis=1), np.ones(1), atol=1e-6)

    def test_bias_matches_step_width(self) -> None:
        x = np.ones((1, 2, 3))
        weight = np.array([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            attention_scores(x, weight, bias=np.zeros(3))

    def test_masked_softmax_epsilon_on_all_masked_row(self) -> None:
        scores = np.array([[1.0, 2.0]])
        mask = np.array([[0.0, 0.0]])
        weights = masked_softmax(scores, mask=mask, epsilon=1e-6)
        self.assertFalse(np.isnan(weights).any())


class ClassifyTests(unittest.TestCase):
    def test_logreg_separates_obvious_data(self) -> None:
        x = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        y = np.array([0, 0, 1, 1])
        weight, bias = fit_logreg(x, y, steps=400)
        pred = predict_label(x, weight, bias)
        np.testing.assert_array_equal(pred, y)
        self.assertEqual(accuracy(y, pred), 1.0)
        self.assertEqual(f1_score(y, pred), 1.0)

    def test_f1_zero_when_no_true_positives(self) -> None:
        self.assertEqual(f1_score(np.array([0, 0]), np.array([0, 0])), 0.0)


class ToyPipelineTests(unittest.TestCase):
    def test_pipeline_module_runs(self) -> None:
        from examples.toy_pipeline import features_for, load_docs, stratified_split

        docs, labels = load_docs()
        word, both = features_for(docs)
        train, test = stratified_split(labels)
        self.assertEqual(word.shape, (16, DIM))
        self.assertEqual(both.shape, (16, DIM * 2))
        self.assertEqual(labels.shape, (16,))
        self.assertEqual(int((labels[train] == 1).sum()), 6)
        self.assertEqual(int((labels[test] == 1).sum()), 2)
        weight, bias = fit_logreg(both[train], labels[train], steps=250)
        pred = predict_label(both[test], weight, bias)
        self.assertEqual(pred.shape, (4,))
        self.assertTrue(set(pred).issubset({0, 1}))


class JsonSmokeTests(unittest.TestCase):
    def test_overview_json_roundtrip(self) -> None:
        payload = json.dumps([summarize_split("subtest")])
        row = json.loads(payload)[0]
        self.assertEqual(row["split"], "subtest")
        self.assertEqual(row["n"], 278)


if __name__ == "__main__":
    unittest.main()
