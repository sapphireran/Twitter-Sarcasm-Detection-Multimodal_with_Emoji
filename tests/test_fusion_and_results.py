from __future__ import annotations

import unittest

import numpy as np

from examples.fusion import (
    build_shared_embedding_matrix,
    pool_corpus,
    pool_tweet,
    toy_tables,
)
from examples.results_catalog import (
    ALL_CARDS,
    BI_LSTM,
    CONDITIONS,
    RANDOM_FOREST,
    SVM,
    ascii_table,
    best_card,
    emoji_gain,
    markdown_table,
)
from examples.tokenize import tokenize_corpus


class FusionTests(unittest.TestCase):
    def test_no_emoji_means_zero_channel(self) -> None:
        words, emoji = toy_tables(dim=4)
        pooled = pool_tweet("I love monday mornings", words, emoji, 4)
        self.assertEqual(pooled.fused_dim, 8)
        self.assertEqual(float(np.linalg.norm(pooled.emoji_mean)), 0.0)
        self.assertGreater(float(np.linalg.norm(pooled.word_mean)), 0.0)
        self.assertEqual(pooled.emoji_hits, [])

    def test_emoji_channel_nonzero_when_present(self) -> None:
        words, emoji = toy_tables(dim=4)
        pooled = pool_tweet("I love monday mornings 😒", words, emoji, 4)
        self.assertIn("😒", pooled.emoji_hits)
        self.assertGreater(float(np.linalg.norm(pooled.emoji_mean)), 0.0)

    def test_corpus_shapes(self) -> None:
        words, emoji = toy_tables(dim=4)
        single, fused = pool_corpus(
            ["I love monday", "Great day 😃"], words, emoji, 4
        )
        self.assertEqual(single.shape, (2, 4))
        self.assertEqual(fused.shape, (2, 8))

    def test_shared_matrix_fallback_fills_emoji(self) -> None:
        words, emoji = toy_tables(dim=4)
        docs = tokenize_corpus(["hello 😒", "I love monday"])
        _, matrix_word, fb_word = build_shared_embedding_matrix(
            docs, words, emoji, 4, use_emoji_fallback=False
        )
        _, matrix_we, fb_we = build_shared_embedding_matrix(
            docs, words, emoji, 4, use_emoji_fallback=True
        )
        self.assertEqual(fb_word, [])
        self.assertIn("😒", fb_we)
        self.assertGreater(float(np.linalg.norm(matrix_we)), float(np.linalg.norm(matrix_word)))


class ResultsCatalogTests(unittest.TestCase):
    def test_condition_count(self) -> None:
        self.assertEqual(len(CONDITIONS), 4)
        for card in ALL_CARDS:
            self.assertEqual(len(card.accuracy), 4)
            self.assertEqual(len(card.f1), 4)

    def test_lstm_beats_forest_on_test(self) -> None:
        self.assertGreater(BI_LSTM.accuracy[0], RANDOM_FOREST.accuracy[0])
        self.assertGreater(BI_LSTM.accuracy[1], RANDOM_FOREST.accuracy[1])

    def test_svm_loses_accuracy_on_full_test_with_emoji(self) -> None:
        gain = emoji_gain(SVM, "accuracy")
        self.assertLess(gain["test_gain"], 0.0)
        self.assertGreater(gain["subtest_gain"], 0.0)

    def test_lstm_subtest_gain_larger_than_test_gain(self) -> None:
        gain = emoji_gain(BI_LSTM, "accuracy")
        self.assertGreater(gain["subtest_gain"], gain["test_gain"])

    def test_best_test_multimodal_is_lstm(self) -> None:
        winner = best_card(ALL_CARDS, "accuracy", 1)
        self.assertEqual(winner.model, "Bi-LSTM + attention")

    def test_tables_include_model_name(self) -> None:
        text = ascii_table(ALL_CARDS, "accuracy")
        md = markdown_table(ALL_CARDS, "f1")
        self.assertIn("Random forest", text)
        self.assertIn("| Bi-LSTM + attention |", md)

    def test_missing_metric_raises(self) -> None:
        from examples.results_catalog import GRADIENT_BOOSTING

        with self.assertRaises(KeyError):
            GRADIENT_BOOSTING.as_row("precision")


if __name__ == "__main__":
    unittest.main()
