from __future__ import annotations

import unittest

from examples.lib.tweet_tokenize import (
    extract_emoji,
    extract_hashtags,
    is_elongated,
    strip_commas_like_readopen,
    tokenize_tweet,
)


class TweetTokenizeTests(unittest.TestCase):
    def test_comma_step_matches_readopen(self) -> None:
        raw = '"So many useless classes , great to be student"'
        cleaned = strip_commas_like_readopen(raw)
        self.assertNotIn(",", cleaned)
        self.assertIn("classes", cleaned)
        self.assertIn("great", cleaned)

    def test_hashtag_and_emoji_stay_atomic(self) -> None:
        tokens = tokenize_tweet(
            "I loovee when people text back ... 😒 #sarcastictweet"
        )
        self.assertIn("#sarcastictweet", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("loovee", tokens)
        self.assertIn("...", tokens)

    def test_user_placeholder(self) -> None:
        tokens = tokenize_tweet("<user> Rest in peace & love")
        self.assertEqual(tokens[0], "<user>")

    def test_lowercase(self) -> None:
        tokens = tokenize_tweet("Don't #SarcasticTweet")
        self.assertIn("#sarcastictweet", tokens)
        self.assertTrue(all(t == t.lower() for t in tokens if t.isalpha()))

    def test_extractors(self) -> None:
        tokens = tokenize_tweet("hate this 😑 #not 😃")
        self.assertEqual(extract_hashtags(tokens), ["#not"])
        self.assertEqual(set(extract_emoji(tokens)), {"😑", "😃"})

    def test_elongation(self) -> None:
        self.assertTrue(is_elongated("soooo"))
        self.assertTrue(is_elongated("loovee"))
        self.assertFalse(is_elongated("love"))
        self.assertFalse(is_elongated("book"))


if __name__ == "__main__":
    unittest.main()
