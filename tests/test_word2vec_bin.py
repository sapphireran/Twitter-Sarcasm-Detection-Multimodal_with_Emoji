from pathlib import Path

import numpy as np
import pytest

from examples.lib.io import repo_root
from examples.lib.word2vec_bin import cosine, header_only, load_word2vec_binary, mean_in_vocab


def test_twitter_emoji_table_header() -> None:
    path = repo_root() / "emoji2vec_twitter.bin"
    vocab_size, dim = header_only(path)
    assert (vocab_size, dim) == (1661, 200)


def test_original_emoji_table_is_300d() -> None:
    vocab_size, dim = header_only(repo_root() / "emoji2vec.bin")
    assert (vocab_size, dim) == (1661, 300)


def test_round_trip_small_table(tmp_path: Path) -> None:
    tokens = ["hello", "😂", "world"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "toy.bin"
    with path.open("wb") as handle:
        handle.write(f"{len(tokens)} {vectors.shape[1]}\n".encode("utf-8"))
        for token, row in zip(tokens, vectors):
            handle.write(token.encode("utf-8") + b" ")
            handle.write(row.astype("<f4").tobytes())

    table = load_word2vec_binary(path)
    assert list(table.tokens) == tokens
    assert table.dim == 3
    np.testing.assert_allclose(table["😂"], vectors[1])
    assert cosine(table["hello"], table["world"]) == pytest.approx(0.0)
    mean = mean_in_vocab(table, ["hello", "missing", "world"])
    np.testing.assert_allclose(mean, np.array([0.5, 0.0, 0.5], dtype=np.float32))


def test_shipped_table_contains_common_faces() -> None:
    table = load_word2vec_binary(repo_root() / "emoji2vec_twitter.bin")
    for glyph in ("😂", "😒", "😍"):
        assert glyph in table
        assert table[glyph].shape == (200,)
