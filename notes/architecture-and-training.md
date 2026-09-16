# Architecture and training notes

Personal commentary on the three Python modules and how I actually used them. This is not an API guide.

## The deep stack in one paragraph

`PrepModel` is a frozen 200-d embedding, two stacked bidirectional LSTMs (256 units each direction), dropout, Raffel-style attention, and a sigmoid. Binary cross-entropy, Adam at 1e-3. I used the same constructor for W and WE. The only difference is how `Preprocess` fills `embedding_matrix`.

I wanted something that could notice a polarity flip late in the tweet ("this is fine" → 🙃) without inventing a second encoder. Attention over the second BiLSTM is the cheap way to let the model pick the face or the punchline instead of averaging the whole 78 steps.

## Why this and not a transformer

Spring 2023, course deadline, one laptop. Fine-tuning BERT would have been the stronger paper move and a worse personal-compute move. GloVe Twitter already speaks this dialect. emoji2vec is the same width, so I could splice it into the table without changing `input_dim`. That splice *is* the multimodal experiment.

If I did this again I would still keep a frozen-embedding recurrent baseline. I would add one pretrained transformer run so I knew the ceiling.

## Embedding table construction

`data_utils.Preprocess` is the important function. Walk-through:

1. Fit a Keras `Tokenizer` on the already-tweet-tokenized train docs.
2. Convert to integer sequences, pad at the end, remember `maxlen` (78 on the saved graphs).
3. Allocate `embedding_matrix` as `(count, 200)` where `count` is the number of train *rows*. See the rant in `dataset-notes.md`. It is large enough, it is still the wrong quantity.
4. For each `word, i` in `tokenizer.word_index`:
   - GloVe hit → copy 200-d vector.
   - Else run `emoji.emoji_list(word)` and keep characters that `emoji.is_emoji`.
   - If that list is non-empty and `get_emoji2vec` is true, mean-pool emoji2vec rows into slot `i`.
   - Anything else, including emoji2vec `KeyError`, becomes zeros. I swallowed exceptions and incremented `nf`. I never printed `nf` in the surviving notebooks.

`get_emoji2vec=False` is the W setting: same code path, but emoji fallback writes zeros instead of the mean. Tokens that GloVe already knows (rare; GloVe Twitter does contain some emoji) stay GloVe even in the W run. So W is not "strip emoji," it is "do not *impute* emoji with emoji2vec."

I like that this keeps the sequence aligned. The model still sees a timestep for 😂. In W that timestep is often a zero vector; in WE it is a point in the same 200-d space as "love" and "hate." That is a much gentler multimodal story than concatenating a 400-d bag.

## Baseline features are a different experiment

`AverageVectorPerTweet` / `AverageVectorPerEmoji` mean-pool independently, then `ml_read_data` concatenates. Order is gone. A tweet that is 15 words and one face becomes one 400-d point. That is why I do not treat "forest WE − forest W" as the same quantity as "BiLSTM WE − BiLSTM W." Same extra resource, different geometry.

Zero-vector policy: if no token hits the relevant vocab, the mean is `zeros(200)`. A tweet with no emoji is `[glove; 0]` on the WE side. The forest can in principle learn "when the second half is ~0, ignore it." SVM's full-test regression suggests it did not always do that.

Shuffle: `ml_read_data` permutes X and y with one index array and applies it to both views. Good. It does **not** take a seed. Retraining baselines will not match the pickled decision surfaces even with identical data.

## LSTM details I actually chose

```text
Bidirectional(LSTM(
    256,
    kernel_initializer='he_normal',
    recurrent_activation='sigmoid',
    return_sequences=True,
    activation='tanh',
))
```

He-normal on the input kernel, tanh / sigmoid gates — hard-sigmoid vs sigmoid was a TF version landmine I do not want to relitigate. Two layers because one felt thin and three felt like a night of OOM. 256 was "the biggest that fit comfortably." Dropout 0.25 after the embedding, 0.4 after each recurrent block. No recurrent dropout. No layer norm.

`trainable=False` on the embedding. I did not want 1.2M GloVe rows wandering on 40k tweets. The WE advantage then has to come from *which* rows are non-zero at init, not from those rows being free to leave the emoji2vec neighborhood.

The SavedModel summaries show every Keras-2 layer wrapped as `module_wrapper_*`. That is a TF 2 reload artifact, not a second architecture. Parameter count 2,510,848 matches two BiLSTM-256 layers plus attention plus dense. The embedding weights are in the graph but marked non-trainable in code; the summary I printed said all 2.51M were trainable, which I think is the wrapper lying about the frozen table. I did not verify.

## Attention layer

`attention_layer.Attention` is the common Keras-2 port of Raffel et al. 2015 (feed-forward attention over time):

```text
e_t = tanh(h_t · W + b_t)
a_t = softmax(e)
context = Σ_t a_t h_t
```

`W` is a vector, not a matrix — one score per timestep. Bias is length `steps`, which is a little unusual (usually scalar or per-feature). It ties the pad length into the layer weights. Retraining with a different `maxlen` will not restore these weights cleanly.

Masking: `supports_masking = True`, and `call` multiplies `exp(e)` by the mask before renormalizing. I still return `None` from `compute_mask`, so nothing downstream sees a mask. Fine here; the next layer is Dense.

Epsilon in the softmax denominator is the NaN guard from the original snippet. I left it.

I did not visualize attention weights. That is the first plot I would add if I reopened the project. If attention does not sit on the emoji (or on the inverted adjective) in the WE model, the +1 acc story is less charming.

## Training procedure I only half recorded

There is no `model.fit` cell in the three notebooks that remain. Training happened elsewhere — a scratch notebook or a script I did not commit. What I can infer:

- Optimizer: Adam `lr=0.001` (the old kwarg; modern Keras wants `learning_rate`).
- Loss: `binary_crossentropy`.
- Metric: `acc`.
- Snapshots named with both test acc and subtest acc, so I was watching both slices while saving.
- Two survivors: W at 0.8635 / 0.8669, WE at 0.8735 / 0.8921.

I do not have epoch count, batch size, early stopping patience, or seed. The evaluate notebook sets `X_val = X_test`, which is a confession that test was on the screen during model selection.

If I retrain, I would: hold out 2k from train as val, fix a seed, save on val F1, and only then touch test and subtest.

## Dropout and the two-layer bet

I used heavy dropout because the train set is 40k short informal sentences and the recurrent stack is not small. The WE model having *higher* test loss and *higher* acc is consistent with a model that is more decisive (precision 0.90) and a bit worse calibrated. I would log BCE and Brier next time instead of staring at acc.

## What I would change in the code, not just the protocol

1. Size the embedding matrix as `vocab_size + 1`.
2. Pass `maxlen` into `Attention` or use a scalar bias so pad length is not baked into `b`.
3. Replace the bare `except:` in `Preprocess` with `KeyError` / `ValueError`.
4. Seed the permutation in `ml_read_data`.
5. Stop constructing `SVC()` in the forest / tree / boosting except-blocks.
6. Write `variables/` when I save a Keras model, or switch to `.keras` / H5 and commit that on purpose.
7. One GloVe loader, one filename, in a tiny `paths.py` so the three notebooks stop disagreeing.

None of that should be required to *read* the 2023 numbers. All of it would be required to *trust* a 2026 rerun.
