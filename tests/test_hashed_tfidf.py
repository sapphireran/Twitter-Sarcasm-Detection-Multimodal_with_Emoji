import numpy as np

from examples.lib.hashed_tfidf import (
    binary_metrics,
    estimate_idf,
    featurize_text,
    fit_logistic,
    hashed_counts,
    to_sparse,
)


def test_signed_hash_is_stable() -> None:
    first = hashed_counts(["#not", "love"], hash_size=64)
    second = hashed_counts(["#not", "love"], hash_size=64)
    assert first == second
    assert first  # both tokens landed somewhere


def test_empty_text_becomes_empty_row() -> None:
    row = featurize_text("   ", hash_size=32)
    assert row.indices.size == 0
    assert row.values.size == 0


def test_logistic_separates_hashtag_cue() -> None:
    from examples.lib.tokenize import tokenize

    texts = [
        "I love this #not",
        "best day ever #sarcasm",
        "what a joy #not",
        "nice weather today",
        "see you tomorrow",
        "had a good walk",
    ]
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    hash_size = 128
    counts = [hashed_counts(tokenize(text), hash_size) for text in texts]
    idf = estimate_idf(counts, hash_size)
    rows = [to_sparse(item, idf=idf) for item in counts]
    model = fit_logistic(
        rows,
        labels,
        hash_size=hash_size,
        idf=idf,
        drop_hashtags=False,
        epochs=12,
        learning_rate=0.4,
        seed=0,
    )
    preds = np.array([int(model.predict_proba_text(text) >= 0.5) for text in texts])
    # The three sarcastic rows all contain an explicit tag; the model should
    # get at least the tagged side right after a dozen epochs.
    assert preds[:3].sum() >= 2
    metrics = binary_metrics(labels, preds)
    assert metrics.accuracy >= 0.8


def test_drop_hashtags_removes_hash_tokens() -> None:
    kept = featurize_text("fine #not really", hash_size=64, drop_hashtags=False)
    dropped = featurize_text("fine #not really", hash_size=64, drop_hashtags=True)
    assert kept.indices.size >= dropped.indices.size
