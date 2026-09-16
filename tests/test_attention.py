import numpy as np
import pytest

from examples.lib.attention import temporal_attention


def test_weights_are_a_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5, 8))
    weight = rng.normal(size=(8,))
    bias = rng.normal(size=(5,))
    context, weights = temporal_attention(x, weight, bias=bias)
    assert context.shape == (3, 8)
    assert weights.shape == (3, 5)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_mask_zeroes_a_timestep_and_renormalises() -> None:
    x = np.ones((1, 3, 2))
    weight = np.array([1.0, 0.0])
    mask = np.array([[1.0, 0.0, 1.0]])
    _, weights = temporal_attention(x, weight, mask=mask)
    assert weights[0, 1] == pytest.approx(0.0)
    assert weights[0, 0] == pytest.approx(weights[0, 2])
    assert weights[0].sum() == pytest.approx(1.0)


def test_constant_sequence_returns_that_vector() -> None:
    hidden = np.array([0.25, -1.5, 3.0])
    x = np.broadcast_to(hidden, (1, 6, 3)).copy()
    weight = np.array([4.0, -2.0, 0.5])
    context, _ = temporal_attention(x, weight)
    np.testing.assert_allclose(context[0], hidden, atol=1e-9)


def test_cue_aligned_weight_puts_mass_on_the_cue() -> None:
    filler = np.array([1.0, 0.0, 0.0])
    cue = np.array([0.0, 1.0, 0.0])
    x = np.stack([np.stack([filler, filler, cue, filler])], axis=0)
    weight = np.array([0.0, 3.0, 0.0])
    context, weights = temporal_attention(x, weight)
    assert int(np.argmax(weights[0])) == 2
    # Two filler steps still contribute to axis 0; the cue axis should
    # dominate the *other* feature and receive the single largest weight.
    assert context[0, 1] > context[0, 2]
    assert weights[0, 2] > 0.4
