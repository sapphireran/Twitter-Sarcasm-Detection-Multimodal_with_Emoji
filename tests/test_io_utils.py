import unittest

from examples.lib.io_utils import SPLIT_FILES, load_all_splits, load_split, repo_root


class IoUtilsTests(unittest.TestCase):
    def test_repo_root_finds_dataset(self) -> None:
        root = repo_root()
        self.assertTrue((root / "dataset" / "train_sentence.csv").exists())

    def test_known_split_sizes(self) -> None:
        expected = {"train": 39780, "test": 2000, "subtest": 278}
        for name, size in expected.items():
            split = load_split(name)
            self.assertEqual(len(split), size)
            self.assertEqual(len(split.labels), size)
            self.assertTrue(set(split.labels) <= {0, 1})

    def test_subtest_is_subset_of_test(self) -> None:
        test = load_split("test")
        subtest = load_split("subtest")
        self.assertTrue(set(subtest.texts) <= set(test.texts))

    def test_load_all_matches_registry(self) -> None:
        splits = load_all_splits()
        self.assertEqual(set(splits), set(SPLIT_FILES))

    def test_unknown_split_raises(self) -> None:
        with self.assertRaises(KeyError):
            load_split("dev")


if __name__ == "__main__":
    unittest.main()
