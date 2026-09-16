import unittest

from ccs2lab.hashed_logreg import featurize, fit_logreg
from ccs2lab.metrics import binary_scores


class HashedLogregTests(unittest.TestCase):
    def test_featurize_length(self) -> None:
        with_cues = featurize("hello #not", hash_dim=32, use_cues=True)
        without = featurize("hello #not", hash_dim=32, use_cues=False)
        self.assertEqual(with_cues.shape, (42,))
        self.assertEqual(without.shape, (32,))

    def test_fits_obvious_not_tag(self) -> None:
        texts = [f"normal tweet {i}" for i in range(20)] + [
            f"wow great #not {i}" for i in range(20)
        ]
        labels = [0] * 20 + [1] * 20
        model = fit_logreg(texts, labels, hash_dim=64, epochs=20, seed=1)
        preds = model.predict_texts(texts)
        scores = binary_scores(labels, preds.tolist())
        self.assertGreaterEqual(scores.accuracy, 0.9)
        cue_w = dict(model.top_cue_weights())
        self.assertGreater(cue_w["not_tag"], 0.0)


if __name__ == "__main__":
    unittest.main()
