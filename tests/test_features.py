from __future__ import annotations

import unittest

import _path  # noqa: F401
from lib.features import FEATURE_NAMES, extract_features, has_leak_hashtag


class FeatureTests(unittest.TestCase):
    def test_leak_hashtag_detection(self):
        self.assertTrue(has_leak_hashtag("body you're doing great #not"))
        self.assertTrue(has_leak_hashtag("#YeahRight sure"))
        self.assertFalse(has_leak_hashtag("note to self: buy milk"))

    def test_include_leak_can_be_forced_off(self):
        text = "I love waiting #not"
        on = extract_features(text, include_leak=True)
        off = extract_features(text, include_leak=False)
        self.assertEqual(on["has_leak_hashtag"], 1)
        self.assertEqual(off["has_leak_hashtag"], 0)
        self.assertEqual(on["has_pos_flip_word"], 1)
        self.assertEqual(on["has_neg_situation"], 1)

    def test_feature_names_are_stable(self):
        feats = extract_features("hello")
        self.assertEqual(list(feats), list(FEATURE_NAMES))
        self.assertTrue(all(v in (0, 1) for v in feats.values()))

    def test_emoji_and_mention(self):
        feats = extract_features("<user> wow 😒")
        self.assertEqual(feats["has_user_mention"], 1)
        self.assertEqual(feats["has_sarcastic_emoji"], 1)
        self.assertEqual(feats["has_any_emoji"], 1)


if __name__ == "__main__":
    unittest.main()
