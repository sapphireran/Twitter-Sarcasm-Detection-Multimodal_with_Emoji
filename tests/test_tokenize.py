import unittest

from examples.tokenize import (
    is_emoji_token,
    normalize_line,
    read_label_file,
    read_pairs,
    tweet_tokenize,
)
from examples.toy_corpus import illustrative_pair


class TokenizeTests(unittest.TestCase):
    def test_comma_join_matches_readopen(self):
        self.assertEqual(normalize_line("foo,bar, baz"), "foo bar  baz")

    def test_hashtag_and_emoji_stay_whole(self):
        tokens = tweet_tokenize("I love Mondays 😒 #Not")
        self.assertEqual(tokens, ["i", "love", "mondays", "😒", "#not"])

    def test_user_placeholder(self):
        tokens = tweet_tokenize("<user> thanks for the notes")
        self.assertEqual(tokens[0], "<user>")

    def test_url_stays_one_token(self):
        tokens = tweet_tokenize("see https://example.com/a now")
        self.assertIn("https://example.com/a", tokens)

    def test_emoji_predicate(self):
        self.assertTrue(is_emoji_token("💉"))
        self.assertTrue(is_emoji_token("😒"))
        self.assertFalse(is_emoji_token("#not"))
        self.assertFalse(is_emoji_token("love"))

    def test_illustrative_pair_token_diff_is_emoji_and_hashtag(self):
        sarcastic, literal = illustrative_pair()
        extra = set(tweet_tokenize(sarcastic)) - set(tweet_tokenize(literal))
        self.assertIn("😒", extra)
        self.assertIn("#not", extra)

    def test_read_label_file_skips_blank_lines(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "y.csv"
            path.write_text("0\n\n1\n0\n", encoding="utf-8")
            self.assertEqual(read_label_file(path), [0, 1, 0])

    def test_read_pairs_rejects_length_mismatch(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            sent = Path(raw) / "s.csv"
            lab = Path(raw) / "y.csv"
            sent.write_text("hello\n", encoding="utf-8")
            lab.write_text("0\n1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_pairs(sent, lab)


if __name__ == "__main__":
    unittest.main()
