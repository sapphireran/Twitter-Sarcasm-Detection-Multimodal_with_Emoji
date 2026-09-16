import unittest

from sarcasm_lib.tokenize import hashtags, tokenize_tweet


class TokenizerTests(unittest.TestCase):
    def test_keeps_mentions_hashtags_and_placeholder(self) -> None:
        tokens = tokenize_tweet(
            "<user> I love walking to school #Not 😄",
            lowercase=True,
        )
        self.assertIn("<user>", tokens)
        self.assertIn("#not", tokens)
        self.assertIn("😄", tokens)
        self.assertIn("love", tokens)

    def test_keeps_url_as_one_token(self) -> None:
        tokens = tokenize_tweet("see https://example.com/a?b=1 please")
        self.assertIn("https://example.com/a?b=1", tokens)

    def test_hashtag_helper(self) -> None:
        tokens = tokenize_tweet("Apartment shopping is so fun #not #exhausted")
        self.assertEqual(hashtags(tokens), ["#not", "#exhausted"])

    def test_lone_hash_is_not_a_hashtag(self) -> None:
        tokens = tokenize_tweet("see # and then #not")
        self.assertEqual(hashtags(tokens), ["#not"])

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize_tweet(""), [])

    def test_lowercase_can_be_disabled(self) -> None:
        tokens = tokenize_tweet("Yay #Not", lowercase=False)
        self.assertIn("Yay", tokens)
        self.assertIn("#Not", tokens)


if __name__ == "__main__":
    unittest.main()
