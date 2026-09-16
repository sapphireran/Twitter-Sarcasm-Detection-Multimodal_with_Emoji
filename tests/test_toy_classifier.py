import unittest

import numpy as np

from examples.average_vectors import multimodal_features
from examples.hash_embeddings import default_tables
from examples.tokenize import tweet_tokenize
from examples.toy_classifier import fit_logreg, metrics, predict, predict_proba
from examples.toy_corpus import TOY_LABELS, TOY_TWEETS, illustrative_pair


class ToyClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text, emoji = default_tables(dim=32)
        tokenized = [tweet_tokenize(t) for t in TOY_TWEETS]
        cls.x_text, cls.x_multi = multimodal_features(tokenized, text, emoji)
        cls.y = np.asarray(TOY_LABELS, dtype=int)
        cls.text_model = fit_logreg(cls.x_text, cls.y, seed=1)
        cls.multi_model = fit_logreg(cls.x_multi, cls.y, seed=1)

    def test_separable_toy_set_fits(self):
        pred = predict(self.multi_model, self.x_multi)
        stats = metrics(self.y, pred)
        self.assertGreaterEqual(stats["accuracy"], 0.875)

    def test_multimodal_separates_illustrative_pair_more(self):
        sarcastic, literal = illustrative_pair()
        i_s = TOY_TWEETS.index(sarcastic)
        i_l = TOY_TWEETS.index(literal)
        gap_text = predict_proba(self.text_model, self.x_text)[i_s] - predict_proba(
            self.text_model, self.x_text
        )[i_l]
        gap_multi = predict_proba(self.multi_model, self.x_multi)[i_s] - predict_proba(
            self.multi_model, self.x_multi
        )[i_l]
        self.assertGreater(gap_multi, gap_text)
        self.assertGreater(gap_multi, 0.15)

    def test_metrics_on_known_confusion(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        stats = metrics(y_true, y_pred)
        self.assertAlmostEqual(stats["accuracy"], 0.75)
        self.assertAlmostEqual(stats["precision"], 1.0)
        self.assertAlmostEqual(stats["recall"], 0.5)
        self.assertAlmostEqual(stats["f1"], 2 / 3)

    def test_loss_decreases(self):
        self.assertLess(self.multi_model.losses[-1], self.multi_model.losses[0])
