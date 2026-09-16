import unittest

from ccs2lab.cues import cue_rule_label, profile_text
from ccs2lab.metrics import binary_scores
from ccs2lab.splits import load_bundle


class CueRuleSplitTests(unittest.TestCase):
    def test_hashtag_rule_beats_recorded_svm_on_test(self) -> None:
        bundle = load_bundle()
        preds = [cue_rule_label(profile_text(text)) for text in bundle.test.texts]
        scores = binary_scores(bundle.test.labels, preds)
        self.assertAlmostEqual(scores.accuracy, 0.8075)
        self.assertEqual(scores.precision, 1.0)
        self.assertGreater(scores.accuracy, 0.769)

    def test_explicit_coverage_test(self) -> None:
        bundle = load_bundle()
        hits = sum(profile_text(text).has_explicit for text in bundle.test.texts)
        self.assertEqual(hits, 615)


if __name__ == "__main__":
    unittest.main()
