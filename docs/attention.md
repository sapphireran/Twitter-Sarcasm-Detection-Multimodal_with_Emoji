# Attention layer

`attention_layer.py` implements a Keras 2 `Layer` for temporal attention
over RNN outputs. The comment in the file points at Raffel & Ellis,
*Feed-Forward Networks with Attention Can Solve Some Long-Term Memory
Problems* (arXiv:1512.08756).

The examples replay the same equations in NumPy so you can see the weights
without loading TensorFlow.

## Shapes

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `x` | `(batch, steps, features)` | BiLSTM output, `return_sequences=True` |
| `W` | `(features,)` | learned score vector |
| `b` | `(steps,)` or omitted | per-timestep bias |
| `e` | `(batch, steps)` | tanh scores |
| `α` | `(batch, steps, 1)` | softmax weights |
| output | `(batch, features)` | weighted sum over time |

`build()` asserts `len(input_shape) == 3`. `compute_mask` returns `None`,
so padding masks stop here.

## Forward pass

From `Attention.call`:

```
e_ij = squeeze(x · expand_dims(W))     # (B, T)
e_ij = e_ij + b                        # if bias
e_ij = tanh(e_ij)
a    = exp(e_ij)
a    = a * cast(mask)                  # if a mask is provided
a    = a / (sum(a, axis=1) + epsilon)
c    = sum(x * expand_dims(a), axis=1) # (B, F)
```

Properties you can test without Keras:

1. `α` is non-negative and sums to 1 along time (within float error).
2. If one timestep is masked, its weight is 0 and the others re-normalise.
3. If `x` is constant across time, `c` equals that constant vector
   regardless of `W` (the weights still sum to 1).
4. A large positive score on one step pulls `c` toward that step's hidden
   state.

`examples/lib/attention.py` and `tests/test_attention.py` encode those
four checks.

## Why the bias is length `steps`

Most textbook Bahdanau attention uses a scalar (or feature-wise) bias
inside `v^T tanh(W h)`. This layer's bias is **time-aligned**: one
learned scalar per position up to the training `maxlen`.

That has two consequences:

- The layer **does not** generalise to a different sequence length. The
  saved models were built with `maxlen` taken from the training pad
  (`78` in the evaluation summary). `preprocess_test` must reuse that
  length.
- The model can learn a positional preference ("pay more attention near
  the end") even before it looks at `W`. For tweets, that is not
  ridiculous: sarcasm tags often arrive last. It is still a limitation if
  you want to reuse the layer on a different pad length.

## Masking

The mask is applied **after** `exp`, which is the usual "masked softmax"
trick: masked positions become 0 in the unnormalised weights, then the
remaining mass is renormalised. The code casts the mask to `K.floatx()`
to avoid float64 promotion (a Theano-era comment that is still correct).

`epsilon` is added to the softmax denominator. Without it, an all-masked
or all-very-negative row can produce NaNs.

## Relation to the rest of the net

`PrepModel` places `Attention()` after the second bidirectional LSTM, so
`features = 512` (256 forward + 256 backward). The dense classifier then
sees one 512-d tweet vector, not a sequence.

This is pooling, not decoder attention. There is no extra query from a
second sequence. The "query" is the learned `W`.

## NumPy walkthrough

`examples/04_attention_walkthrough.py` builds a 4-step, 3-d toy sequence
where step 3 is a sarcasm-cue vector and the others are filler. It prints:

- raw scores `e`
- softmax weights
- the pooled output
- the same run with step 3 masked, to show the mass moving onto the
  remaining steps

Run it after installing `requirements-examples.txt` (NumPy only).
