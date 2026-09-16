import unittest

import _paths  # noqa: F401

from sarcasm_lib.metrics import accuracy, binary_f1
from sarcasm_lib.naive_bayes import MultinomialNB


class NaiveBayesTests(unittest.TestCase):
    def test_separates_hashtag_supervision(self) -> None:
        texts = [
            "great day #not",
            "love this weather #sarcasm",
            "great day with friends",
            "love this weather today",
            "best news ever #yeahright",
            "best news ever honestly",
        ]
        labels = [1, 1, 0, 0, 1, 0]
        model = MultinomialNB(min_count=1, include_cues=True)
        model.fit(texts, labels)
        pred = model.predict(texts)
        self.assertEqual(pred, labels)
        self.assertGreaterEqual(accuracy(labels, pred), 1.0)

    def test_strip_tags_removes_leak(self) -> None:
        texts = [
            "same words #not",
            "same words",
            "same words #sarcasm",
            "same words again",
        ]
        labels = [1, 0, 1, 0]
        leaky = MultinomialNB(min_count=1, strip_supervision_tags=False)
        leaky.fit(texts, labels)
        hidden = MultinomialNB(min_count=1, strip_supervision_tags=True)
        hidden.fit(texts, labels)
        self.assertIn("#not", leaky.vocab_)
        self.assertNotIn("#not", hidden.vocab_)
        self.assertNotIn("#sarcasm", hidden.vocab_)

    def test_predict_proba_rows_sum_to_one(self) -> None:
        texts = ["happy birthday", "happy birthday #not"]
        labels = [0, 1]
        model = MultinomialNB(min_count=1)
        model.fit(texts, labels)
        proba = model.predict_proba(["happy birthday #not", "happy birthday"])
        self.assertEqual(proba.shape, (2, 2))
        self.assertTrue(((proba.sum(axis=1) - 1.0) ** 2 < 1e-12).all())

    def test_f1_on_held_out_phrase(self) -> None:
        train_x = ["tired again #not", "tired again after work", "fun times", "fun times #sarcasm"]
        train_y = [1, 0, 0, 1]
        model = MultinomialNB(min_count=1)
        model.fit(train_x, train_y)
        pred = model.predict(["tired again #not", "fun times"])
        self.assertGreater(binary_f1([1, 0], pred), 0.0)


if __name__ == "__main__":
    unittest.main()
