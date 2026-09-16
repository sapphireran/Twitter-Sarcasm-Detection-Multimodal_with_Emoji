# Architecture

Two code paths share embeddings and diverge at the encoder.

```
tweet text
    │
    ├─ classical ─ TweetTokenizer ─ per-token GloVe lookup
    │                    │
    │                    ├─ mean → 200-d ─ SVM / DT / RF / GBT
    │                    └─ mean(GloVe) ⊕ mean(emoji2vec) → 400-d
    │
    └─ neural ─ Keras Tokenizer ─ padded id sequence (length L ≈ 78)
                     │
                     └─ frozen Embedding(V, 200)
                            │
                            ├─ single-modal: GloVe row or zeros
                            └─ multi-modal: GloVe row, else mean emoji2vec
                                   │
                                   Dropout 0.25
                                   BiLSTM(256)  →  (B, L, 512)
                                   Dropout 0.4
                                   BiLSTM(256)  →  (B, L, 512)
                                   Dropout 0.4
                                   Attention    →  (B, 512)
                                   Dense(1, sigmoid)
```

`L = 78` is the padded length observed when the saved models were
summarised in `evaluate_loaded_dl_models.ipynb`. It is a function of the
training set, not a constant in `dl_model.py`. `PrepModel` takes `l` from
`Preprocess`.

## Embedding matrix construction

`data_utils.Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True)`:

1. Fit a Keras `Tokenizer` on the already-tokenised training tweets.
2. Convert texts to integer sequences and `pad_sequences(..., padding='post')`.
3. Allocate `embedding_matrix` with shape `(count, 200)`. `count` is the
   number of **training tweets**, which is an upper bound on the vocabulary
   plus one, not `len(tokenizer.word_index) + 1`. Rows that are never
   touched stay zero.
4. For each `word, i` in `tokenizer.word_index`:
   - if `word` is in GloVe, copy that 200-d vector;
   - else collect emoji codepoints inside `word` with `emoji.emoji_list`;
   - if any, average their emoji2vec rows when `get_emoji2vec` is true;
   - otherwise write zeros.

The single-modal neural model is the same function with `get_emoji2vec=False`
(or a call that still walks emoji but writes zeros). Either way, GloVe rows
are frozen (`trainable=False` in `PrepModel`).

`preprocess_test` reuses the training tokenizer and the training `maxlen` so
the test tensor lines up with the embedding matrix.

## Classical vector construction

`AverageVectorPerTweet` / `AverageVectorPerEmoji` walk tokens and average
every in-vocabulary hit. An empty hit list becomes a zero vector of length
200. `ml_read_data` then concatenates the two averages for the multi-modal
matrix.

Gensim 4 renamed `KeyedVectors.vocab` to `key_to_index`. The 2023 helpers
still use `.vocab` and `model[word]`. A current Gensim install will need
those lookups rewritten (`word in model` / `model[word]` still works;
`.vocab` does not). The example emoji reader does not use Gensim.

## Neural model (`dl_model.PrepModel`)

```
Embedding(count, 200, weights=embedding_matrix, input_length=l, trainable=False)
Dropout(0.25)
Bidirectional(LSTM(256, he_normal, tanh, recurrent sigmoid, return_sequences=True))
Dropout(0.4)
Bidirectional(LSTM(256, same))
Dropout(0.4)
Attention()
Dense(1, sigmoid)
Adam(lr=0.001), binary cross-entropy, accuracy
```

Parameter count reported in the evaluation notebook: **2,510,848**, all
trainable according to the summary. That summary is slightly misleading.
The embedding weights were created as non-trainable, but the loaded
SavedModel wraps several layers in `ModuleWrapper`, and the printed
"trainable params" line counts those wrappers as trainable. Trust the
source in `dl_model.py` over the wrapper summary: the 200-d embedding
table was intended to stay frozen.

`Adam(lr=...)` is the Keras 2 / TF 2.10-era keyword. Current Keras wants
`learning_rate`.

## Attention

The custom layer is a one-layer feed-forward scorer over time:

```
e_t = tanh(h_t · W + b_t)
α   = softmax(e)
c   = Σ_t α_t h_t
```

`W` has shape `(features,)`. `b` has shape `(steps,)` when bias is on, so
it is a **per-timestep** bias, not a scalar. Masked timesteps are zeroed
after `exp` and before the softmax normalisation. A small `epsilon` is
added to the denominator so early training does not divide by zero.

This is the Raffel & Ellis (2015) "feed-forward attention" used as a
differentiable pooling head, not multi-head self-attention.

Math, masking, and a NumPy replay: [`attention.md`](attention.md) and
`examples/04_attention_walkthrough.py`.

## Why stacked BiLSTMs plus attention

The course framing was: a tweet is a short sequence with a possible late
cue (`#not` at the end, an emoji after a positive adjective). A
unidirectional LSTM can bury the opening clause; a bidirectional one sees
both ends; attention can put weight on the cue without forcing the last
hidden state to carry everything.

The classical models test the cheaper alternative: ignore order, keep the
same embeddings. Random forest winning that group (81.5–81.8% test
accuracy) is consistent with "hashtags + a few lexical polarity flips are
strong features." The extra 5–6 points from the BiLSTM are the order-
sensitive remainder.

## Saved artifacts

| Path | What the notebooks treat it as |
| --- | --- |
| `model/best_model_single_modal` | BiLSTM + attention, GloVe only |
| `model/best_model_multi_modal` | same, GloVe + emoji2vec OOV fill |
| `baseline_models/*.pkl` | sklearn dumps (not present in this checkout) |

`get_metrics_of_models.ipynb` also refers to longer directory names that
embed the scores (`best_model_we_0.8734...`). Those names are from an
older layout; the evaluation notebook that still runs relative to this
tree uses the shorter `model/best_model_*` paths.
