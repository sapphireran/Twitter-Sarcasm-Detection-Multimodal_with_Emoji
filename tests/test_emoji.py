import unittest

from sarcasm_lib.emoji import count_emojis, extract_emojis, has_emoji


class EmojiTests(unittest.TestCase):
    def test_extracts_repeated_faces(self) -> None:
        self.assertEqual(extract_emojis("pretty please?! 😭 😭 😭"), ["😭", "😭", "😭"])

    def test_has_emoji(self) -> None:
        self.assertTrue(has_emoji("I love watching golf ! ⛳"))
        self.assertFalse(has_emoji("plain text only"))

    def test_counts_across_texts(self) -> None:
        counts = count_emojis(["hi 😒", "bye 😒 😍"])
        self.assertEqual(counts["😒"], 2)
        self.assertEqual(counts["😍"], 1)


if __name__ == "__main__":
    unittest.main()
