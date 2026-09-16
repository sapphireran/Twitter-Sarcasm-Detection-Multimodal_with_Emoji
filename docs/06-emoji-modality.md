# 06 — The emoji channel

## What “multimodal” means in this repo

There is no image encoder. A tweet is a string that happens to contain
Unicode pictographs. Those pictographs are mapped through **emoji2vec**
(Eisner, Rocktäschel, Augenstein, Bošnjak, Riedel 2016): a skip-gram style
table in which an emoji is trained to land near the words in its name and
description (“unamused face”, “red heart”, …).

Two files are committed:

| File | Shape | Role |
| --- | --- | --- |
| `emoji2vec_twitter.bin` | 1,661 × 200 | Used by the 2023 notebooks with GloVe-Twitter 200d |
| `emoji2vec.bin` | 1,661 × 300 | Upstream 300-d table; not used by `data_utils.py` |

Confirm with `python3 examples/04_inspect_emoji2vec.py`. The word2vec
binary header is `1661 200` / `1661 300`.

## How a tweet becomes two vectors

`AverageVectorPerTweet` averages every token that hits GloVe. Emoji
usually miss GloVe (they are not in the 27B Twitter vocab as isolated
code points in the same form), so the W vector is a word-only mean.

`AverageVectorPerEmoji` averages every token that hits emoji2vec. Words
miss, faces hit. All-miss → 200 zeros.

WE for sklearn is `concat(word_mean, emoji_mean)` → 400-d.

The neural path is stricter: one embedding row per tokenizer id. A token
is *either* a GloVe row *or* an emoji average *or* zeros. A tweet like
“love 😒” therefore has “love” as GloVe and “😒” as emoji2vec in the
**same** 200-d sequence, which is what the BiLSTM sees. That is a
cleaner fusion than concatenation-of-means, and it is why the net can
attend to the face.

## Coverage on this dataset

`examples/04_inspect_emoji2vec.py` reports type- and document-frequency
coverage of train pictographs against `emoji2vec_twitter.bin`. Expect:

* The common faces (😂 😍 ❤ 😭 😒 😊) are in-vocab.
* Combined sequences (ZWJ families, some flag / skin-tone clusters) are
  the bulk of OOV. `data_utils.Preprocess` wraps the emoji branch in a
  bare `except:` and writes zeros, so those tokens are silently dropped
  from the emoji channel.
* HEART without a variation selector (`❤`, U+2764, df=428 in train) does
  not match emoji2vec’s fully-qualified `❤️` (U+2764 + U+FE0F).
  `lookup_emoji` in `examples.common.word2vec` tries both; a raw
  `token in model.vocab` check does not. The 2023 `emoji` package
  typically emits the fully-qualified sequence, so the notebooks may still
  have hit this row.

A high OOV rate on rare emoji is acceptable: they never had enough
support to move a 40k-tweet model anyway.

## Co-occurrence (before embeddings)

`examples/02_emoji_cooccurrence.py` computes document-frequency PMI between
each pictograph and the sarcastic class. Qualitatively, on train:

* 😂 is the most common face in **both** classes. It is a weak cue.
* 😒 and 😑 have higher sarcastic rate (deadpan / “this is fine”).
* ❤ and 😍 lean literal (affection, unironic praise).
* 😭 appears in both “I am actually sad” and hyperbolic sarcasm.

PMI is not a model. It is the reason a 200-d table *could* help: the
pictograph is not independent of the label. The subtest filter (100%
emoji) is how you let that dependence show up in a metric.

## Failure modes that look like “emoji didn’t help”

1. **Concatenating zeros on 86% of test rows.** RBF SVM got *worse* on
   full test when going W → WE (76.9 → 76.3). Extra constant dimensions
   plus a kernel that cares about scale will do that. The net, which
   shares the 200-d space, does not have that problem in the same way.
2. **Cue hashtags dominate.** If `#sarcasm` is still in the string,
   attention may never need the face. Strip hashtags before claiming an
   emoji-only story (`examples/06_bow_baseline.py --` `stripped` mode).
3. **Mislabeled or ironic-without-being-sarcastic rows.** A crying face
   on a literal sad tweet is not a sarcasm cue. See
   [07-annotated-examples.md](07-annotated-examples.md).
4. **Frozen emoji2vec vs. this domain.** Unicode descriptions are not
   Twitter. “Fire” 🔥 on Twitter is often “this is great”, not a
   combustion event. A description-trained table can point the wrong
   way. That is an untested limitation of the 2023 setup.

## Design choice worth keeping

Evaluating on an **emoji-only slice** is the right way to talk about this
channel. If you add a new fusion method (learned gate, emoji CNN, even a
one-hot emoji flag), report test *and* subtest, W *and* WE. The 2023
table already has that shape; do not collapse it to a single accuracy.
