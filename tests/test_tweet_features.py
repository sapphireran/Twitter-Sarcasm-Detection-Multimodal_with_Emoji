import json
import unittest
from pathlib import Path

from examples.lib.tweet_features import (
    extract_emojis,
    extract_features,
    normalize_commas,
    tokenize_tweet,
)

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "sample_tweets.json"


class TweetFeatureTests(unittest.TestCase):
    def test_commas_become_spaces(self) -> None:
        self.assertEqual(
            normalize_commas("So many useless classes , great to be student"),
            "So many useless classes   great to be student",
        )

    def test_tokenizer_keeps_hashtags_mentions_and_emoji(self) -> None:
        tokens = tokenize_tweet("Don't you love it #not 😃 🔫 <user> 100 days")
        self.assertIn("#not", tokens)
        self.assertIn("<user>", tokens)
        self.assertIn("😃", tokens)
        self.assertIn("🔫", tokens)
        self.assertIn("love", tokens)
        self.assertIn("100", tokens)
        self.assertNotIn("1", tokens)

    def test_sarcastic_sample_has_contrast_cues(self) -> None:
        features = extract_features("I just love having grungy ass hair 😑 #not")
        self.assertEqual(features.not_hashtag, 1)
        self.assertEqual(features.love_word, 1)
        self.assertEqual(features.contrast_cue, 1)
        self.assertGreaterEqual(features.emoji_count, 1)
        self.assertGreaterEqual(features.negative_emoji_count, 1)

    def test_condolence_is_not_a_flip(self) -> None:
        features = extract_features("<user> Rest in peace & love to you and your family")
        self.assertEqual(features.not_hashtag, 0)
        self.assertEqual(features.sarcasm_hashtag, 0)
        self.assertEqual(features.contrast_cue, 0)
        self.assertEqual(features.love_word, 1)

    def test_sincere_crying_love_is_not_contrast(self) -> None:
        features = extract_features("i wanna love you again live someday 😭 😭 😭")
        self.assertEqual(features.love_word, 1)
        self.assertEqual(features.contrast_cue, 0)
        self.assertGreaterEqual(features.emoji_count, 3)

    def test_elongated_love(self) -> None:
        playful = extract_features("I loovee when people text back ... 😒 #sarcastictweet")
        self.assertEqual(playful.love_word, 1)
        self.assertEqual(playful.sarcasm_hashtag, 1)
        self.assertEqual(playful.ellipsis, 1)
        self.assertEqual(playful.elongated_count, 0)

        stretched = extract_features("I loooove this #not")
        self.assertEqual(stretched.elongated_count, 1)
        self.assertEqual(stretched.love_word, 1)

    def test_extract_emojis(self) -> None:
        self.assertEqual(extract_emojis("ok 😭 😭 😭"), ["😭", "😭", "😭"])

    def test_sample_file_is_well_formed(self) -> None:
        payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(payload["tweets"]), 8)
        for tweet in payload["tweets"]:
            self.assertIn(tweet["label"], (0, 1))
            self.assertTrue(tweet["text"].strip())
            extract_features(tweet["text"])


if __name__ == "__main__":
    unittest.main()
