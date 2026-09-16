import unittest

from ccs2lab.sample import stratified_take
from ccs2lab.splits import load_bundle


class SampleTests(unittest.TestCase):
    def test_prefix_of_train_is_almost_all_sincere(self) -> None:
        bundle = load_bundle()
        prefix = bundle.train.labels[:12000]
        self.assertLess(sum(prefix), 20)

    def test_stratified_take_keeps_rate(self) -> None:
        texts = ["s"] * 100 + ["z"] * 80
        labels = [0] * 100 + [1] * 80
        taken_t, taken_y = stratified_take(texts, labels, 36, seed=3)
        self.assertEqual(len(taken_t), 36)
        rate = sum(taken_y) / len(taken_y)
        self.assertGreater(rate, 0.3)
        self.assertLess(rate, 0.6)

    def test_limit_zero_returns_all(self) -> None:
        texts = ["a", "b"]
        labels = [0, 1]
        out_t, out_y = stratified_take(texts, labels, 0)
        self.assertEqual(out_t, texts)
        self.assertEqual(out_y, labels)


if __name__ == "__main__":
    unittest.main()
