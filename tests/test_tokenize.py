from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.tokenize import is_emoji_token, is_hashtag_token, tokenize_tweet, whitespace_len


class TokenizeTests(unittest.TestCase):
    def test_hashtag_and_emoji_stay_whole(self):
        tokens = tokenize_tweet("I love this 😑 #Not")
        self.assertIn("#not", tokens)
        self.assertIn("😑", tokens)
        self.assertIn("love", tokens)
        self.assertNotIn("#", tokens)

    def test_user_placeholder(self):
        tokens = tokenize_tweet("<user> hello @Alice")
        self.assertIn("<user>", tokens)
        self.assertIn("@alice", tokens)

    def test_url(self):
        tokens = tokenize_tweet("see https://example.com/a now")
        self.assertTrue(any(tok.startswith("https://") for tok in tokens))

    def test_empty(self):
        self.assertEqual(tokenize_tweet(""), [])
        self.assertEqual(whitespace_len("   "), 0)

    def test_lowercase_flag(self):
        self.assertEqual(tokenize_tweet("Hi #NOT", lowercase=False), ["Hi", "#NOT"])

    def test_ellipsis_stays_whole(self):
        self.assertIn("...", tokenize_tweet("text back ... 😒"))

    def test_variation_selector_is_dropped(self):
        # ❤ often arrives as HEART + VS16; keep the pictograph only.
        tokens = tokenize_tweet("love ❤\ufe0f now")
        self.assertIn("❤", tokens)
        self.assertNotIn("\ufe0f", tokens)

    def test_classifiers(self):
        self.assertTrue(is_hashtag_token("#not"))
        self.assertFalse(is_hashtag_token("#"))
        self.assertTrue(is_emoji_token("😒"))
        self.assertFalse(is_emoji_token("\ufe0f"))


if __name__ == "__main__":
    unittest.main()
