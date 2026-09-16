# Architecture

The project is two stacks that share tokenization and then diverge.

```text
CSV tweets + labels
        │
        ▼
  ReadOpen (TweetTokenizer, lowercase)
        │
        ├──────────── classical ─────────────┐
        │                                    │
        ▼                                    ▼
 mean GloVe 200-d                    mean GloVe ⊕ mean emoji2vec
        │                                    │
        ▼                                    ▼
 SVM / DT / RF / GBT                 same four models, 400-d input
        │
        └──────────── deep ──────────────────┐
                                             ▼
                              Keras Tokenizer + post-padding
                                             ▼
                              200-d embedding (frozen)
                                ├ GloVe for words
                                └ emoji2vec for emoji tokens (multimodal)
                                             ▼
                              Dropout 0.25
                                             ▼
                              BiLSTM 256  → 512-d sequence
                                             ▼
                              Dropout 0.4
                                             ▼
                              BiLSTM 256  → 512-d sequence
                                             ▼
                              Dropout 0.4
                                             ▼
                              Attention (Raffel) → 512-d tweet vector
                                             ▼
                              Dense(1, sigmoid) + Adam + BCE
```

## Classical stack

Implemented in `baseline_models.ipynb` on top of `ml_read_data`.

Each classifier is trained twice:

- `*_classifier.pkl` — 200-d word averages
- `*_classifier_we.pkl` — 400-d word+emoji averages

Default sklearn constructors were used (`SVC()`, `DecisionTreeClassifier()`, `RandomForestClassifier()`, `GradientBoostingClassifier()`). There is no grid search in the notebook. That keeps the comparison honest as a “same hyperparameters, extra features?” test, and it also means none of these models is tuned to a local optimum.

Random forest is the strongest classical model in the recorded numbers. That matches the representation: a forest can split on a few large-magnitude embedding directions (including emoji dimensions 200–399) without assuming linear separability.

## Deep stack

Implemented in `dl_model.PrepModel(count, embedding_matrix, l, lrate=0.001)`.

```text
Embedding(count, 200, weights=embedding_matrix, input_length=l, trainable=False)
Dropout(0.25)
Bidirectional(LSTM(256, return_sequences=True,
                   kernel_initializer='he_normal',
                   activation='tanh',
                   recurrent_activation='sigmoid'))
Dropout(0.4)
Bidirectional(LSTM(256, return_sequences=True, …))
Dropout(0.4)
Attention()
Dense(1, activation='sigmoid')
```

Compiled with `Adam(lr=0.001)`, `binary_crossentropy`, and `acc`.

The saved multimodal model summary (from `evaluate_loaded_dl_models.ipynb`) reports:

- sequence length 78
- bidirectional outputs of size 512
- about **2.51M** parameters, all marked trainable in that printout because the embedding was wrapped in a `ModuleWrapper` during the TF2 load

In the *construction* code the embedding is frozen. If you reload an old SavedModel, check `model.layers[0].trainable` before you continue training.

### Why bidirectional + attention

Tweets are short enough that a unidirectional LSTM would already see the whole sentence, but sarcasm often hinges on a late cancel (`… #not`) or an early setup (`Don't you love it when…`). Bidirectional states let the cancel token see the setup, and vice versa. Attention then builds a weighted sum of the 78 time steps so the classifier is not forced to read only the last hidden state.

A 512-d attended vector feeding a single sigmoid is a small head. Almost all of the work is in the two BiLSTM layers.

## Single-modal vs multimodal, again

| | Classical | Deep |
| --- | --- | --- |
| How emoji enter | Extra 200 dimensions concatenated after averaging | Same 200-d table; emoji tokens get emoji2vec rows |
| Word/emoji alignment | None (bag + bag) | Positional (sequence) |
| Order | Discarded | Kept until attention pools |

This is why a *small* multimodal gain on the balanced test set and a *larger* gain on the emoji-heavy subtest is the expected pattern, not a contradiction.

## Attention layer

`attention_layer.py` is a Keras 2-era layer. Documented separately in [`attention.md`](attention.md). The examples folder includes a NumPy reimplementation you can run without TensorFlow: `examples/attention_walkthrough.py`.

## Training loop (as used in 2023)

The architecture file only **builds** the model. Fitting happened in notebooks that are not fully preserved as a clean `model.fit(...)` script. The evaluation notebooks load `model/best_model_single_modal` and `model/best_model_multi_modal`. Treat those directories as the official artifacts; treat `PrepModel` as the recipe that produced them.

If you retrain:

- keep `return_sequences=True` on **both** LSTMs or `Attention` will receive the wrong rank
- keep the same tokenizer and `maxlen` as the checkpoint, or the embedding rows will not line up
- do not change `count` (the first dimension of the embedding matrix) after the tokenizer is frozen
