# Architecture

Two modeling stacks share the same tweet files and the same two
pretrained embedding families. They differ in how a tweet becomes a
vector and in how emoji enter that vector.

```
                    ┌───────────── W path ──────────────┐
 tweets ──tokenize──┤  GloVe lookup                     ├── classifier
                    │  emoji → 0 (classical) or OOV row │
                    └───────────── WE path ─────────────┘
                                 GloVe lookup
                                 emoji → emoji2vec
```

## Tokenization (shared)

Both stacks start from `ReadOpen`:

1. Strip the raw line.
2. Replace commas with spaces (`split(',')` then `' '.join`).
3. `nltk.TweetTokenizer` (preserves hashtags, emoji, emoticons).
4. Lowercase every token.

That is why `#Not` and `#not` are the same feature, and why `<user>`
survives as a single token when it appears in train.

A standalone replay of this pipeline (without NLTK) lives in
`examples/preprocess_walkthrough.py`. The fallback tokenizer is a
regex approximation used only for the walkthrough; the original
experiments used NLTK.

## Classical models (W vs WE)

`ml_read_data` mean-pools token vectors.

```
W  : mean({ GloVe(t) | t in tweet ∩ GloVe.vocab })          → R^200
E  : mean({ emoji2vec(t) | t in tweet ∩ emoji2vec.vocab })  → R^200
WE : concat(W, E)                                           → R^400
```

Empty intersections become zeros. A tweet with words but no emoji
therefore has a real W half and a zero E half. Concatenation still
gives the classifier a chance to learn "emoji slot is empty".

Classifiers, all sklearn defaults unless noted:

| Name | Estimator | Input |
| --- | --- | --- |
| SVM | `SVC()` | W or WE |
| Decision Tree | `DecisionTreeClassifier()` | W or WE |
| Random Forest | `RandomForestClassifier()` | W or WE |
| Gradient Boosting | `GradientBoostingClassifier()` | W or WE |

The notebook cells that *train* RF / GBT / DT have a copy-paste bug:
some `except FileNotFoundError` branches fit `SVC()` for the W model
and the intended estimator only for WE. The **recorded metrics** come
from the successfully loaded pickles, not from those fallback cells.
If you retrain from scratch, fix those branches first.

## Deep model (Bi-LSTM + attention)

`PrepModel` in `dl_model.py`:

```
tokens (padded to L=78)
        │
        ▼
Embedding(V, 200, trainable=False)     frozen GloVe / emoji2vec table
        │
        ▼
Dropout(0.25)
        │
        ▼
BiLSTM(256) → sequences                (B, L, 512)
        │
        ▼
Dropout(0.4)
        │
        ▼
BiLSTM(256) → sequences                (B, L, 512)
        │
        ▼
Dropout(0.4)
        │
        ▼
Attention (Raffel 2015)                (B, 512)
        │
        ▼
Dense(1, sigmoid)
```

Compiled with `Adam(lr=0.001)`, `binary_crossentropy`, metric `acc`.

### How WE is injected

There is no second encoder. The embedding table is filled as:

| Token | W table (`get_emoji2vec=False`) | WE table (`get_emoji2vec=True`) |
| --- | --- | --- |
| in GloVe | GloVe row | GloVe row |
| OOV but contains emoji | zeros | mean of emoji2vec rows |
| other OOV | zeros | zeros |

So the LSTM sees emoji as real 200-d tokens only in the WE run. In
the W run those positions are zero vectors (plus whatever bias the
position-specific attention term learns).

### Attention

`attention_layer.Attention` scores each timestep independently:

```
e_t = tanh(h_t · w + b_t)
α   = softmax(e)
c   = Σ_t α_t h_t
```

`w ∈ R^{512}` is shared across time. `b ∈ R^{L}` is **not** — it is a
learned bias per position. After `build()` the layer is tied to the
padded length (78 in the saved models). Masking is supported: padded
steps can be zeroed after the exponential, then renormalized with
`epsilon` to avoid NaNs.

`examples/attention_demo.py` runs this formula on a 4-step toy
sequence so you can print `e`, `α`, and the context vector without
Keras.

## Why two test sets

| Set | What it measures |
| --- | --- |
| test (2,000, balanced) | Overall sarcasm detection, including tweets with no emoji |
| subtest (278, all non-ASCII) | Whether the emoji channel moves the needle when emoji are actually present |

A model can win on test with hashtags and contrastive wording alone.
The subtest WE−W gap is the project's evidence that emoji2vec is not
just dead weight.

## Training / evaluation flow (original)

```
glove.twitter.27B.200d.bin ─┐
emoji2vec_twitter.bin      ─┤
dataset/*_sentence.csv     ─┼─► data_utils ─► sklearn pickles
dataset/*_label.csv        ─┤                 or Keras SavedModel
random seed (unset)        ─┘
```

`ml_read_data` calls `np.random.permutation` with no seed. Classical
metrics are therefore not bitwise-reproducible across re-shuffles
even with the same pickles, because the notebook also shuffles
**test**. The numbers in [results.md](results.md) are the values
printed in the committed notebook outputs, not a fresh run.

The deep notebooks do **not** shuffle after `Preprocess`, so test
order stays file order.
