import unittest

from ccs2lab.tokenize import csv_line_to_text, hashtags, tokenize_tweet


class TokenizeTests(unittest.TestCase):
    def test_csv_commas_become_spaces(self) -> None:
        self.assertEqual(csv_line_to_text("a,b,c"), "a b c")

    def test_keeps_hashtags_mentions_and_user(self) -> None:
        tokens = tokenize_tweet("<user> love this #Not @bob")
        self.assertIn("<user>", tokens)
        self.assertIn("#not", tokens)
        self.assertIn("@bob", tokens)
        self.assertIn("love", tokens)

    def test_keeps_multidigit_numbers(self) -> None:
        tokens = tokenize_tweet("100 days until Christmas")
        self.assertIn("100", tokens)
        self.assertNotIn("1", tokens)

    def test_keeps_emoji_and_apostrophes(self) -> None:
        tokens = tokenize_tweet("I'm sooo done 😒")
        self.assertIn("i'm", tokens)
        self.assertIn("😒", tokens)

    def test_hashtags_helper(self) -> None:
        tokens = tokenize_tweet("wow #sarcasm #not ready")
        self.assertEqual(hashtags(tokens), ["#sarcasm", "#not"])

    def test_url_stays_together(self) -> None:
        tokens = tokenize_tweet("see https://example.com/x now")
        self.assertTrue(any(tok.startswith("https://") for tok in tokens))


if __name__ == "__main__":
    unittest.main()
