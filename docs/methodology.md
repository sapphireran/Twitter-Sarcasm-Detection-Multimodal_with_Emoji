# Methodology

This note records the research framing of the 2023 UCPH CCS2 project. It is written from the code and the executed notebooks in this repository, not from a recovered PDF write-up.

## Task

Binary classification:

- Input: one English tweet, already lightly normalized (`<user>` for mentions, commas later turned into spaces).
- Output: `1` if the tweet is sarcastic, `0` otherwise.

Sarcasm here is operationalized the way the source labels operationalize it. Many positive examples are explicit (`#not`, `#sarcastictweet`, `#yeahright`). That makes the task easier than fully implicit sarcasm, but it is also closer to how people actually mark sarcasm on Twitter.

## Why a second modality

A unimodal word embedding sees the lexical content. It is weak on three cues that show up constantly in this dataset:

1. **Emoji that invert polarity.** `I love walking to school 😄` is ambiguous until the hashtag `#SarcasticTweet` or a weary face lands on the same line.
2. **Emoji that are the only non-literal signal.** `I just love getting shots 💉` uses a literal object emoji next to a fake-positive verb.
3. **Zero lexical overlap with training sarcasm templates.** Classical averages collapse word order. Emoji still give the classifier a sparse, stable handle.

The second channel is [emoji2vec](https://arxiv.org/abs/1609.08359) (Eisner et al.), trained so that emoji sit in the same space as the words they co-occur with. This repository ships `emoji2vec.bin` and a Twitter-tuned `emoji2vec_twitter.bin`. The notebooks load the Twitter file.

Text uses Stanford GloVe Twitter 27B, 200 dimensions, converted to word2vec binary so `gensim.KeyedVectors` can read it. That file is **not** in git (about 1 GB). The examples package substitutes a tiny hash embedding so the math can be exercised without it.

## Two fusion recipes

The project does not learn a cross-attention mixer. It uses two early-fusion recipes that are easy to ablate.

### Recipe A — concatenate averages (sklearn baselines)

For tweet tokens \(t_1, \ldots, t_n\):

\[
\mathbf{x}_{\text{text}} = \frac{1}{|\{t_i \in V_{\text{glove}}\}|} \sum_{t_i \in V_{\text{glove}}} \mathbf{e}_{\text{glove}}(t_i)
\]

\[
\mathbf{x}_{\text{emoji}} = \frac{1}{|\{t_i \in V_{\text{emoji}}\}|} \sum_{t_i \in V_{\text{emoji}}} \mathbf{e}_{\text{emoji}}(t_i)
\]

If a tweet has no in-vocabulary tokens in a channel, that channel is a **200-d zero vector**. The multimodal feature is \([\mathbf{x}_{\text{text}}; \mathbf{x}_{\text{emoji}}] \in \mathbb{R}^{400}\). The single-modal ablation drops the emoji half.

This is exactly `AverageVectorPerTweet` + `AverageVectorPerEmoji` + `np.concatenate` in `data_utils.py`.

### Recipe B — mixed embedding table (Bi-LSTM)

`Preprocess` builds one Keras tokenizer over the training tweets and a matrix \(E \in \mathbb{R}^{|V| \times 200}\):

- If token \(w\) is in GloVe, \(E_w \leftarrow\) GloVe.
- Else extract emoji code points from \(w\) with the `emoji` package. If any of those exist in emoji2vec, \(E_w \leftarrow\) their mean.
- Else \(E_w \leftarrow \mathbf{0}\).
- Single-modal ablation: skip the emoji2vec branch (`get_emoji2vec=False`) and write zeros for OOV tokens.

The sequence model therefore still has one 200-d channel, but emoji tokens are no longer unknown. Attention can put weight on `😒` or `💉` the same way it puts weight on `love`.

## Sequence encoder

`PrepModel` in `dl_model.py`:

```
Embedding(vocab, 200, frozen)
  → Dropout(0.25)
  → Bidirectional(LSTM(256, return_sequences=True))
  → Dropout(0.4)
  → Bidirectional(LSTM(256, return_sequences=True))
  → Dropout(0.4)
  → Attention()          # Raffel et al. 2015
  → Dense(1, sigmoid)
```

Optimizer: Adam, default learning rate `0.001`. Loss: binary cross-entropy. Metric: accuracy.

The attention layer is a learned token-wise gate. For hidden states \(h_1, \ldots, h_T \in \mathbb{R}^{512}\) (256 × 2 directions):

\[
e_t = \tanh(h_t^\top w + b_t), \qquad
\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k) + \varepsilon}, \qquad
c = \sum_{t=1}^{T} \alpha_t h_t
\]

`w` is shared across time. `b` is a per-position bias of length \(T\) (the padded length, 78 on the saved models). This matches the Keras layer in `attention_layer.py` and the numpy reimplementation in `examples/attention_numpy.py`.

## Evaluation protocol

Three numbers matter:

1. **Test accuracy / F1** on 2,000 balanced tweets. Headline number.
2. **Subtest accuracy / F1** on 278 tweets that lean sarcastic and emoji-heavy. This is the slice where Recipe A and Recipe B are supposed to help.
3. **Single vs multi** for every model, so a lift cannot be explained by "we trained a bigger classifier."

Notebooks also dump precision and recall. Random forest and the Bi-LSTM both recall sarcasm more aggressively than they precision it on the main test set; the multimodal Bi-LSTM is the exception (precision 0.904 vs recall 0.836 on test). See [results.md](results.md).

## What this is not

- Not a transformer, not a tweet-BERT fine-tune. The 2023 coursework target was "does a cheap second channel help a classical / LSTM stack."
- Not late fusion. There is no learned gate between a text expert and an emoji expert.
- Not a claim about production moderation. Labels include many explicit sarcasm hashtags; implicit literary sarcasm is under-represented.
- Not a fully reproducible training dump. GloVe is missing from git, and several notebook cells assume files in the working directory. The examples package is the reproducible part.

## Design choices that show up in the code

| Choice | Where | Consequence |
| --- | --- | --- |
| `TweetTokenizer` then `.lower()` | `ReadOpen` | Hashtags stay intact (`#not`), emoji stay tokens, URLs usually survive as one piece |
| Join commas to spaces before tokenizing | `ReadOpen` | CSV leftovers do not become empty fields |
| Zero vector for empty channels | `AverageVectorPer*` | Tweets without emoji are still valid 400-d rows; the emoji half is all zeros |
| Frozen embeddings | `PrepModel` | 2.51 M trainable params are almost all LSTM + attention; the 200-d table does not fine-tune |
| `he_normal` + `tanh` / `sigmoid` recurrent | `PrepModel` | Keras 2-era defaults for LSTM numerical stability |
| Shuffle with `np.random.permutation` | `ml_read_data` | Train order is not deterministic across runs unless you seed numpy first |

The examples package copies the averaging, zero-fill, concatenation, and attention equations so they can be unit-tested without Gensim or Keras.
