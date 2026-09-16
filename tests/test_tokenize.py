import unittest

from sarcasm_lib.tokenize import (
    TweetishTokenizer,
    extract_emoji,
    extract_hashtags,
    is_emoji_char,
    tokenize_tweet,
)


class TokenizeTests(unittest.TestCase):
    def test_hashtags_mentions_and_words(self) -> None:
        tokens = tokenize_tweet("<user> Late nights #Not thanks https://x.test")
        self.assertIn("<user>", tokens)
        self.assertIn("#not", tokens)
        self.assertIn("late", tokens)
        self.assertIn("nights", tokens)
        self.assertTrue(any(tok.startswith("http") for tok in tokens))

    def test_emoji_clusters(self) -> None:
        tokens = tokenize_tweet("great job 😂😂 😒")
        self.assertIn("😂😂", tokens)
        self.assertIn("😒", tokens)
        self.assertIn("great", tokens)

    def test_extract_helpers(self) -> None:
        text = "Love this #YeahRight 😊 #not"
        self.assertEqual(extract_hashtags(text), ["#yeahright", "#not"])
        self.assertEqual(extract_emoji(text), ["😊"])

    def test_keep_punct(self) -> None:
        raw = TweetishTokenizer(keep_punct=True, lowercase=False).tokenize("Wow!!!")
        self.assertIn("!", raw)

    def test_is_emoji_char_rejects_ascii(self) -> None:
        self.assertFalse(is_emoji_char("a"))
        self.assertTrue(is_emoji_char("😂"))


if __name__ == "__main__":
    unittest.main()
