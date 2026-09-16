import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ccs2lab.paths import EMOJI2VEC_300, EMOJI2VEC_TWITTER_200
from ccs2lab.w2v import average_present, load_word2vec_bin, read_word2vec_header


def _write_bin(path: Path, items: list[tuple[str, list[float]]]) -> None:
    dim = len(items[0][1])
    with path.open("wb") as handle:
        handle.write(f"{len(items)} {dim}\n".encode("ascii"))
        for token, vec in items:
            handle.write(token.encode("utf-8") + b" ")
            handle.write(struct.pack("<" + "f" * dim, *vec))


class Word2VecTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.bin"
            _write_bin(path, [("😂", [1.0, 0.0]), ("😒", [0.0, 1.0])])
            n, dim = read_word2vec_header(path)
            self.assertEqual((n, dim), (2, 2))
            table = load_word2vec_bin(path)
            self.assertIn("😂", table)
            near = table.most_similar("😂", k=1)
            self.assertEqual(near[0][0], "😒")

    def test_average_present_skips_oov(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.bin"
            _write_bin(path, [("😒", [2.0, 0.0]), ("😭", [0.0, 4.0])])
            table = load_word2vec_bin(path)
            vec = average_present(["hello", "😒", "😭"], table)
            np.testing.assert_allclose(vec, np.array([1.0, 2.0], dtype=np.float32))
            zeros = average_present(["nope"], table)
            np.testing.assert_allclose(zeros, np.zeros(2))

    def test_checked_in_headers(self) -> None:
        self.assertEqual(read_word2vec_header(EMOJI2VEC_TWITTER_200), (1661, 200))
        self.assertEqual(read_word2vec_header(EMOJI2VEC_300), (1661, 300))


if __name__ == "__main__":
    unittest.main()
