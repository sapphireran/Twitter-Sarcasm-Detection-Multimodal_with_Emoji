from __future__ import annotations

import unittest

from examples.lib.emoji_extract import extract_emojis, has_high_codepoint, is_emoji_char
from examples.lib.tweet_tokenize import tokenize_tweet, whitespace_tokens


class TokenizeTests(unittest.TestCase):
    def test_hashtag_and_face(self) -> None:
        tokens = tokenize_tweet("I loovee when people text back ... 😒 #sarcastictweet")
        self.assertIn("#sarcastictweet", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("loovee", tokens)
        self.assertIn("...", tokens)

    def test_user_placeholder(self) -> None:
        tokens = tokenize_tweet("<user> Rest in peace")
        self.assertEqual(tokens[0], "<user>")

    def test_apostrophe(self) -> None:
        tokens = tokenize_tweet("Don't you love it")
        self.assertIn("don't", tokens)

    def test_lowercase(self) -> None:
        tokens = tokenize_tweet("Oh HOW I Love #NOT")
        self.assertIn("#not", tokens)
        self.assertIn("love", tokens)

    def test_whitespace_differs(self) -> None:
        text = "Don't text back ... 😒#not"
        tweet = tokenize_tweet(text)
        ws = whitespace_tokens(text)
        self.assertNotEqual(tweet, ws)
        self.assertIn("don't", tweet)
        self.assertIn("#not", tweet)
        self.assertIn("😒", tweet)


class EmojiTests(unittest.TestCase):
    def test_extract_cluster(self) -> None:
        self.assertEqual(extract_emojis("hi 😒 #not"), ["😒"])
        self.assertIn("🌲", extract_emojis("Christmas! 🌲 soon"))

    def test_glues_spaced_variation_selector(self) -> None:
        self.assertEqual(extract_emojis("Great start ☺ ️ #NOT"), ["☺️"])
        self.assertNotIn("️", extract_emojis("Great start ☺ ️ #NOT"))

    def test_is_emoji_char(self) -> None:
        self.assertTrue(is_emoji_char("😒"))
        self.assertFalse(is_emoji_char("a"))
        self.assertFalse(is_emoji_char("ab"))

    def test_high_codepoint_filter(self) -> None:
        self.assertTrue(has_high_codepoint("I loovee 😒"))
        self.assertFalse(has_high_codepoint("plain ascii #not"))


if __name__ == "__main__":
    unittest.main()
