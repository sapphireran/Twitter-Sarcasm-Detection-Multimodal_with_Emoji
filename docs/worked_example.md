# Worked example: one sarcastic subtest tweet

Take the first subtest line:

```text
I loovee when people text back ... 😒 #sarcastictweet
gold = 1
```

## Example tokenizer (`sarcasm_lab`)

```text
tokens: i | loovee | when | people | text | back | ... | 😒 | #sarcastictweet
cues:   sarcasm-hashtag=#sarcastictweet, neg-emoji, positive-stem, ellipsis×1
rule:   predict 1
```

The distant-supervision tag alone is enough for the cue rule. The positive stem (“loovee when”) plus `😒` is the Riloff-style contrast the 2023 project wanted emoji2vec to see.

## 2023 Keras path (what `Preprocess` would do)

1. NLTK `TweetTokenizer` produces a similar token list (lowercased).
2. Keras `Tokenizer` maps each token to an integer. `#sarcastictweet` and `😒` get distinct ids if they occurred in training.
3. The embedding row for `#sarcastictweet` is Twitter GloVe if that hashtag is in the 27B dump; otherwise zeros.
4. The embedding row for `😒`:
   - **single-modal:** zeros (`get_emoji2vec=False`)
   - **multimodal:** emoji2vec vector for that character (or a mean if several emoji were glued into one token)
5. The padded sequence is length 78. Attention can put weight on the last two real tokens instead of averaging them with “I loovee when”.

That is the entire multimodal story for this tweet: keep a nonzero row for `😒`, and let attention find `#sarcastictweet` / the face instead of mean-pooling them into a 200-D soup.

## Why bag-of-words also gets this one right

Unigram NB / logistic see `#sarcastictweet` as a feature with huge class odds (log-odds ≈ 4.8 on train). They do not need GloVe or an LSTM. They fail on the *next* kind of tweet — verbal irony with no tag — which is exactly the 371 rule false negatives in `examples/error_analysis.py`.
