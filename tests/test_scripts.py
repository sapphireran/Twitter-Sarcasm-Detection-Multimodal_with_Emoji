import unittest

from examples.run_attention import main as attention_main
from examples.run_pipeline import main as pipeline_main
from examples.run_real_tokens import main as real_tokens_main
from examples.run_toy_classifier import main as classifier_main
from examples.tokenize import tweet_tokenize
from examples.dataset_stats import SPLIT_FILES
from examples.tokenize import read_pairs


class ScriptSmokeTests(unittest.TestCase):
    def test_pipeline_main(self):
        self.assertEqual(pipeline_main(), 0)

    def test_attention_main(self):
        self.assertEqual(attention_main(), 0)

    def test_classifier_main(self):
        self.assertEqual(classifier_main(), 0)

    def test_real_tokens_main(self):
        self.assertEqual(real_tokens_main(), 0)

    def test_first_subtest_row_has_emoji_token(self):
        sent, lab = SPLIT_FILES["subtest"]
        text, label = read_pairs(sent, lab)[0]
        tokens = tweet_tokenize(text)
        self.assertEqual(label, 1)
        self.assertTrue(any(ord(tok[0]) > 255 for tok in tokens if tok))
