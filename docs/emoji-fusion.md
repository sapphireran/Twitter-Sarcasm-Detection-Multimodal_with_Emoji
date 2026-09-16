# Emoji fusion

The multimodal claim of this project is narrow: **keep emoji in a vector space that already knows about them, then measure whether sarcasm metrics move.** There is no learned cross-modal attention. There are two places a 200-d emoji vector can enter the pipeline.

## Channel 1 — tweet GloVe

`glove.twitter.27B.200d` is trained on tweets, so a lot of informal tokens (`lol`, `#not`, `<user>`, elongated words) already have rows. Emoji coverage in GloVe Twitter is incomplete and encoding-dependent. A tweet whose only sarcastic cue is `😒` often looks like a slightly positive sentence in the GloVe average.

## Channel 2 — emoji2vec

[Eisner et al., 2016](https://arxiv.org/abs/1609.08359) skip-gram the words that appear near an emoji in a large tweet dump, then place the emoji at the average of those contexts. The result is a 300-d space in the original paper; the binaries in this repo are used as 200-d Twitter-aligned vectors (`emoji2vec_twitter.bin`).

`data_utils.py` never trains emoji2vec. It only does `model_emoji2vec[token]` lookups.

Two binaries are checked in:

| File | Role |
| --- | --- |
| `emoji2vec_twitter.bin` | What the notebooks load |
| `emoji2vec.bin` | Upstream-style dump, unused by the notebooks |

## Fusion A: concatenate means (baselines)

```
x_text   = mean({ glove(t)   for t in tokens if t in glove })
x_emoji  = mean({ emoji2vec(t) for t in tokens if t in emoji2vec })
x_multi  = concat(x_text, x_emoji)          # 400-d
```

Empty sets become zeros, so a tweet with no emoji is `[x_text ; 0]`. The classifier can in principle learn "if the second half is near zero, ignore it." Random forest is the baseline that actually uses that extra half on the subtest (+4.7 acc, +3.1 F1). SVM slightly **hurts** on the main test set when the extra half is added (−0.6 acc), which is the usual "double the features, same n" penalty.

`examples/fusion.py` implements the same concat and a cosine report so you can see that sarcastic toy tweets sit closer to each other in 400-d than in 200-d when they share an emoji and little lexical overlap.

## Fusion B: write emoji rows into \(E\) (Bi-LSTM)

`Preprocess(..., get_emoji2vec=True)`:

```
for word, i in tokenizer.word_index.items():
    if word in glove:
        E[i] = glove[word]
    else:
        chars = emoji code points inside word
        vecs  = [emoji2vec[c] for c in chars if c in emoji2vec]
        if vecs and get_emoji2vec:
            E[i] = mean(vecs)
        else:
            E[i] = 0
```

Two details worth knowing:

1. **GloVe wins ties.** If a token is in GloVe, emoji2vec is never consulted. Fusion B therefore mainly rescues tokens that GloVe missed, not tokens that GloVe already represents poorly.
2. **The `emoji` package is only used on the GloVe-OOV path.** A word like `shots💉` (no space) can still donate `💉` to the matrix. A standalone `💉` token that happens to be in GloVe keeps the GloVe row.

Single-modal ablation is the same function with `get_emoji2vec=False`: OOV rows stay zero. That is how `best_model_single_modal` was trained relative to `best_model_multi_modal`. Both graphs have the same parameter count (2,510,848) because the embedding table is frozen and the LSTM widths do not change. The only difference is what those frozen rows contain.

## Why the subtest moves more than the test set

| Set | Sarcasm rate | Emoji rate | Marker rate | RF / Bi-LSTM acc lift |
| --- | --- | --- | --- | --- |
| test (2,000) | 50% | 13.8% | 30.5% | +0.4 / +1.0 |
| subtest (278) | 62% | **99.3%** | 47.8% | +4.7 / +2.5 |

If the emoji channel were noise, the subtest would not systematically prefer the multimodal column. If it were a universal free lunch, the balanced test set would move by a similar amount. The recorded pattern matches "the second channel is sparse and useful when present."

`examples/inspect_dataset.py` prints the emoji-rate and hashtag-rate gap so this is not an impressionistic claim.

## What fusion does *not* do

- It does not align GloVe and emoji2vec with a trained projection. Concatenation assumes the classifier can learn two different 200-d geometries. The mixed table assumes they are already interchangeable enough to share an LSTM.
- It does not model emoji **order** in the baseline path. `😂 😂 😂` and one `😂` become the same mean.
- It does not special-case skin-tone modifiers or ZWJ sequences beyond whatever `emoji.emoji_list` + `emoji.is_emoji` return.
- It does not use the Unicode name of an emoji (`crying face`) as extra text.

## Toy check

`examples/run_pipeline.py` embeds:

```
i love mondays 😒 #not          (sarcastic)
i love mondays                  (literal-looking)
```

and shows that Fusion A only separates them in the second 200-d half. That is the entire multimodal idea, stripped of Gensim.
