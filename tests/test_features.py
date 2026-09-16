import unittest

from sarcasm_lib.dataset import load_split
from sarcasm_lib.features import (
    SUPERVISION_TAGS,
    cue_flags,
    has_supervision_tag,
    informative_log_odds,
    lexical_tokens,
    strip_supervision_tags,
    summarize_cues,
)
from collections import Counter


class FeatureTests(unittest.TestCase):
    def test_supervision_tag_detection(self) -> None:
        self.assertTrue(has_supervision_tag("wow #NOT"))
        self.assertTrue(has_supervision_tag("sure #sarcastictweet"))
        self.assertFalse(has_supervision_tag("wow not really"))
        stripped = strip_supervision_tags("wow #NOT sure #sarcasm")
        self.assertNotIn("#not", stripped.lower())
        self.assertNotIn("#sarcasm", stripped.lower())

    def test_cue_flags(self) -> None:
        flags = cue_flags("<user> nice job 😒 #yeahright")
        self.assertTrue(flags.has_emoji)
        self.assertTrue(flags.has_user)
        self.assertTrue(flags.has_explicit_sarcasm_tag)
        self.assertFalse(flags.has_not_hashtag)

    def test_lexical_tokens_add_cues(self) -> None:
        tokens = lexical_tokens("hello 😂 #not", include_cues=True)
        self.assertIn("CUE:emoji", tokens)
        self.assertIn("CUE:not_tag", tokens)
        hidden = lexical_tokens("hello 😂 #not", strip_tags=True, include_cues=True)
        self.assertNotIn("CUE:not_tag", hidden)
        self.assertNotIn("#not", hidden)

    def test_log_odds_ranks_indicative_token(self) -> None:
        neg = Counter({"ok": 20, "rare": 1})
        pos = Counter({"ok": 2, "rare": 20})
        rows = informative_log_odds(neg, pos, min_count=5)
        tokens = [row.token for row in rows]
        self.assertEqual(tokens[0], "rare")
        self.assertGreater(rows[0].log_odds, 0)

    def test_subtest_is_emoji_heavy(self) -> None:
        cues = summarize_cues(load_split("subtest"))
        self.assertGreaterEqual(cues.emoji_neg + cues.emoji_pos, 270)
        self.assertGreater(cues.n_pos, cues.n_neg)

    def test_supervision_tag_set_covers_dataset_markers(self) -> None:
        self.assertIn("#not", SUPERVISION_TAGS)
        self.assertIn("#yeahright", SUPERVISION_TAGS)


if __name__ == "__main__":
    unittest.main()
