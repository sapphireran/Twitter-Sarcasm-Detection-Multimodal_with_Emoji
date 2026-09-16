from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.tokenize import tokenize_tweet, whitespace_len


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


if __name__ == "__main__":
    unittest.main()
