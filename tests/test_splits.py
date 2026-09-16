import unittest

from ccs2lab.splits import integrity, load_bundle


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle()
        cls.report = integrity(cls.bundle)

    def test_expected_sizes(self) -> None:
        self.assertEqual(len(self.bundle.train), 39780)
        self.assertEqual(len(self.bundle.test), 2000)
        self.assertEqual(len(self.bundle.subtest), 278)

    def test_labels_are_binary(self) -> None:
        for split in self.bundle.all_splits():
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_test_is_balanced(self) -> None:
        self.assertEqual(self.bundle.test.n_sarcastic, 1000)
        self.assertEqual(self.bundle.test.n_sincere, 1000)

    def test_subtest_is_subset_of_test(self) -> None:
        self.assertTrue(self.report.subtest_subset_of_test)
        self.assertEqual(self.report.test_subtest_overlap, 278)
        self.assertEqual(self.report.train_subtest_overlap, 0)

    def test_train_test_overlap_is_the_known_48(self) -> None:
        self.assertEqual(self.report.train_test_overlap, 48)


if __name__ == "__main__":
    unittest.main()
