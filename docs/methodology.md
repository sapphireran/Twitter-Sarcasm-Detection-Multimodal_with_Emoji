# Methodology notes

How the 2023 personal project turned tweets into single-modal and multi-modal
sarcasm decisions. This is a reconstruction from the checked-in Python, not a
new experiment.

## Two input views

Every tweet is scored twice.

1. **Single-modal (W).** Word embeddings only. GloVe Twitter 200-d
   (`glove.twitter.27B.200d`).
2. **Multi-modal (WE).** The same word vector plus an emoji2vec 200-d vector.
   In the sklearn baselines the two 200-d means are concatenated to 400-d. In
   the LSTM the emoji vector is written *into* the embedding matrix for tokens
   that GloVe missed.

“Multi-modal” here means *text + emoji embeddings*, not image pixels. There is
no vision model.

## Shared preprocessing

`data_utils.ReadOpen`:

1. Read raw lines (not a CSV parser).
2. Replace commas with spaces.
3. Tokenize with NLTK `TweetTokenizer`.
4. Lowercase every token.
5. Load the parallel label file with pandas.

That comma rewrite is easy to miss. Quoted tweets such as

```text
"So many useless classes , great to be student"
```

become a flat token stream with the quotation marks still attached to the
first/last tokens depending on the tokenizer.

## Classical baselines

`ml_read_data` builds one vector per tweet:

- `AverageVectorPerTweet` — mean of GloVe rows for in-vocabulary tokens,
  otherwise a 200-d zero vector.
- `AverageVectorPerEmoji` — mean of emoji2vec rows for in-vocabulary emoji,
  otherwise zeros.
- Multi-modal vector = `concat(word_mean, emoji_mean)`.

The same random permutation is applied to both views so W and WE stay aligned.

Classifiers in `baseline_models.ipynb`:

| Name | sklearn class | Notes |
| --- | --- | --- |
| SVM | `SVC()` | default RBF |
| Decision Tree | `DecisionTreeClassifier()` | the notebook’s fallback path accidentally fits an `SVC` for the WE tree |
| Random Forest | `RandomForestClassifier()` | the fallback path has the same class mix-up |
| Gradient Boosting | `GradientBoostingClassifier()` | same |

Those fallback bugs only matter if the `.pkl` files are missing. In the 2023
run the notebooks loaded already-trained pickles, so the recorded numbers are
still the intended models.

Each classifier is trained once on W features and once on WE features, then
scored on test and subtest.

## Deep model

`dl_model.PrepModel` is a frozen-embedding Bi-LSTM with Raffel-style attention:

```text
Embedding(vocab, 200, trainable=False)
  → Dropout(0.25)
  → Bidirectional(LSTM(256, return_sequences=True))
  → Dropout(0.4)
  → Bidirectional(LSTM(256, return_sequences=True))
  → Dropout(0.4)
  → Attention()          # (batch, steps, 512) → (batch, 512)
  → Dense(1, sigmoid)
```

Optimizer: Adam at `lr=0.001`. Loss: binary cross-entropy.

The embedding matrix comes from `data_utils.Preprocess`:

1. Keras `Tokenizer` on the training token lists.
2. Pad sequences on the right.
3. For each word index:
   - if it is in GloVe, copy that row;
   - else peel emoji codepoints out of the token with the `emoji` package and
     average their emoji2vec rows when `get_emoji2vec=True`;
   - else write zeros.

Single-modal training sets `get_emoji2vec=False` so unknown tokens stay zero
instead of inheriting emoji geometry. Multi-modal training leaves the flag on.

`preprocess_test` reuses the training tokenizer and the training maximum
length. The saved models were trained with length 78.

## Attention layer

`attention_layer.Attention` is the 2016 feed-forward temporal attention:

```text
e_t = tanh(h_t · w + b_t)
α   = softmax(e)
c   = Σ_t α_t h_t
```

`w` has shape `(features,)`. `b` has shape `(steps,)`, so it is a **per-timestep**
bias, not a per-feature bias. Masks are applied after `exp` and the sum is
stabilized with `epsilon`.

`examples/lib/attention_numpy.py` is a TensorFlow-free port of that formula.
`python -m examples.attention_demo` runs it on a 3-token sarcastic sketch
(`love`, `dirty-house`, `#not`) so you can see the flip token take most of the
mass.

## Why emoji can help

Sarcasm in this collection is often a *contrast*: a positive predicate, an
unpleasant situation, then a cue.

```text
I just love having grungy ass hair 😑 #not
```

Word embeddings see `love` and `hair`. The deadpan face and `#not` are the
actual flip. Mean-pooling or a bag of GloVe rows can drown that signal.
Attention and a dedicated emoji channel are two different ways to keep it.

The lexical example in `examples/lexical_baseline_demo.py` uses the same idea
without neural nets: hand-built contrast, `#not`, polarity, and elongation
features.

## What this repo no longer reproduces out of the box

- GloVe Twitter 200-d is not checked in (large).
- Baseline pickles are missing.
- The Keras SavedModel folders only contain `keras_metadata.pb`.

See [reproduction.md](reproduction.md) if you want to rebuild the 2023 path.
The new examples are the path that runs on this clone.
