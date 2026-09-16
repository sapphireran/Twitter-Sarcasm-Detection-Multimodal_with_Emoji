import unittest

from sarcasm_lib.heuristic import predict_sarcasm, score_tweet
from sarcasm_lib.io import load_split
from sarcasm_lib.metrics import binary_metrics


class HeuristicCueTests(unittest.TestCase):
    def test_marker_hashtag_is_enough(self) -> None:
        result = score_tweet("I love walking to school 😄 #not")
        self.assertEqual(result.predicted, 1)
        self.assertGreaterEqual(result.score, 2.0)
        self.assertTrue(any("hashtag" in reason for reason in result.reasons))

    def test_plain_positive_tweet_is_not_sarcastic(self) -> None:
        result = score_tweet("Rest in peace and love to you and your family")
        self.assertEqual(result.predicted, 0)
        self.assertEqual(result.score, 0.0)

    def test_positive_opener_plus_negative_emoji(self) -> None:
        result = score_tweet("I just love getting shots 😒")
        self.assertEqual(result.predicted, 1)
        self.assertGreaterEqual(result.score, 1.0)

    def test_predict_wrapper(self) -> None:
        self.assertEqual(predict_sarcasm("#sarcastictweet of course"), 1)


class HeuristicCorpusSmokeTests(unittest.TestCase):
    def test_beats_majority_on_subtest(self) -> None:
        """The emoji-heavy subtest is full of explicit #not cues."""
        split = load_split("subtest")
        preds = [predict_sarcasm(text) for text in split.texts]
        metrics = binary_metrics(split.labels, preds)
        majority = max(split.positive_count, split.negative_count) / len(split)
        self.assertGreater(metrics.accuracy, majority)
        self.assertGreater(metrics.f1, 0.7)


if __name__ == "__main__":
    unittest.main()
