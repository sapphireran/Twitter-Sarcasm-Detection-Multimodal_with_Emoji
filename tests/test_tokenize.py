from __future__ import annotations

import unittest

from examples.tokenize import join_commas, tokenize_corpus, tokenize_tweet


class TokenizeTests(unittest.TestCase):
    def test_join_commas_matches_readopen(self) -> None:
        raw = "hello,world,again"
        self.assertEqual(join_commas(raw), "hello world again")

    def test_hashtag_and_mention_stay_whole(self) -> None:
        tokens = tokenize_tweet("I love this #Not @alice <user>")
        self.assertIn("#not", tokens)
        self.assertIn("@alice", tokens)
        self.assertIn("<user>", tokens)
        self.assertTrue(all(token == token.lower() or not token.isalpha() for token in tokens))

    def test_emoji_kept(self) -> None:
        tokens = tokenize_tweet("I love monday 😒")
        self.assertIn("😒", tokens)

    def test_adjacent_emoji_stay_separate(self) -> None:
        tokens = tokenize_tweet("ok 😃 🔫")
        self.assertIn("😃", tokens)
        self.assertIn("🔫", tokens)
        self.assertNotIn("😃🔫", tokens)

    def test_url_kept(self) -> None:
        tokens = tokenize_tweet("see https://example.com/a now")
        self.assertTrue(any(token.startswith("https://") for token in tokens))

    def test_lowercase_flag(self) -> None:
        tokens = tokenize_tweet("LoVe", lowercase=False)
        self.assertEqual(tokens, ["LoVe"])

    def test_corpus(self) -> None:
        docs = tokenize_corpus(["Hello #SARCASM", "ok"])
        self.assertEqual(docs[0][-1], "#sarcasm")
        self.assertEqual(docs[1], ["ok"])


if __name__ == "__main__":
    unittest.main()
