import unittest

from ccs2lab.cues import cue_rule_label, profile_text


class CueTests(unittest.TestCase):
    def test_not_tag_is_explicit(self) -> None:
        profile = profile_text("great class #not")
        self.assertTrue(profile.has_not_tag)
        self.assertTrue(profile.has_explicit)
        self.assertEqual(cue_rule_label(profile), 1)

    def test_notes_is_not_a_not_tag(self) -> None:
        profile = profile_text("see my #notes later")
        self.assertFalse(profile.has_not_tag)
        self.assertFalse(profile.has_explicit)

    def test_sincere_love_crying_is_not_contrast(self) -> None:
        profile = profile_text("I love you 😭")
        self.assertTrue(profile.has_positive)
        self.assertTrue(profile.has_negative_emoji)
        self.assertFalse(profile.has_contrast)

    def test_love_plus_deadpan_and_hashtag_is_contrast(self) -> None:
        profile = profile_text("I love when people text back 😒 #sarcastictweet")
        self.assertTrue(profile.has_contrast)
        self.assertTrue(profile.has_sarcasm_tag)

    def test_elongation_ignores_dots(self) -> None:
        dotted = profile_text("wait...")
        self.assertFalse(dotted.has_elongation)
        long_word = profile_text("sooo tired")
        self.assertTrue(long_word.has_elongation)

    def test_cue_rule_negative(self) -> None:
        profile = profile_text("just a normal tuesday")
        self.assertEqual(cue_rule_label(profile), 0)


if __name__ == "__main__":
    unittest.main()
