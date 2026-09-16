# Attention layer

`attention_layer.py` implements the feed-forward temporal attention of [Raffel and Ellis, 2015](https://arxiv.org/abs/1512.08756), adapted as a Keras layer that sits on an RNN with `return_sequences=True`.

## Shapes

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Input `x` | `(batch, steps, features)` | BiLSTM outputs; here `(B, 78, 512)` |
| `W` | `(features,)` | one score vector, not a matrix |
| `b` | `(steps,)` | optional per-timestep bias |
| Output | `(batch, features)` | weighted sum over time |

This is **not** multi-head self-attention. There is no query/key pair between tokens. Each time step is scored independently against the same `W`, then the scores are softmaxed (with a mask) and used as mixture weights.

## Forward pass

For a single example, dropping the batch index:

```text
e_t  = tanh( x_t · W  +  b_t )          # scalar
a_t  = exp(e_t) * mask_t
a    = a / (sum(a) + ε)
h    = sum_t a_t * x_t                  # (features,)
```

The code uses `K.dot(x, K.expand_dims(W))` and then `K.squeeze`, which is the batched form of `x_t · W`. Softmax is written as `exp` / `(sum + epsilon)` so an all-masked or all-near-zero row does not become NaN. After the weights are applied, `compute_mask` returns `None`: the next layer (the Dense head) sees a single vector and should not inherit the sequence mask.

## Why a per-timestep bias?

`b` has shape `(steps,)`, not `(features,)`. It can learn that “the last few tokens of a tweet are more likely to carry the sarcasm hashtag” as a **positional** prior, independent of content. On a dataset where `#not` is usually tweet-final, that prior is easy to pick up. It is also a reason to be cautious: a model that leans on “attend to the end” may degrade if you strip trailing hashtags.

## Masking

`supports_masking = True`. Post-padding means the tail of the 78-step sequence is zeros. If a mask is provided, padded steps get `a_t = 0` before renormalization, so they do not leak into `h`. If a mask is *not* provided, the layer will still attend to pad steps whose `tanh(x_t · W)` happens to be large. When you re-implement this, pass the pad mask through.

## How to read the weights

After training, `W` is a 512-d direction in BiLSTM space. Tokens whose hidden state aligns with `W` get high `e_t`. You can, in principle:

1. run a tweet through the BiLSTM
2. compute `a_t`
3. print the original tokens sorted by `a_t`

That is the most useful interpretability tool this repo has. The 2023 notebooks did not plot those alignments. `examples/attention_walkthrough.py` does the scoring math on a synthetic sequence so the formula is testable without loading Keras.

## Keras 2 vs TensorFlow 2

The docstring says the layer was tested with Keras 2.0.6. The evaluation notebooks load SavedModels through `tf.keras.models.load_model`. Custom layers often need `custom_objects={"Attention": Attention}` on a modern TF2 install; the original notebook omitted that argument and relied on the then-current SavedModel wrapping (`ModuleWrapper` layers in the printed summary). If a reload fails, register the class before calling `load_model`.

`get_config` is not implemented. That means `model.save()` of a *rebuilt* `PrepModel` may drop layer hyperparameters unless you add a config method first.
