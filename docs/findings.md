# Findings from the example scripts

These observations come from running `examples/` on the committed files. They are about **this snapshot of the data**, not a claim about sarcasm detection in general.

## The subtest is an emoji slice

Every subtest tweet contains at least one emoji (`dataset_overview.py`: 100%). Training and the full test set are only ~14% emoji. Combined with the 61.9% sarcasm rate, the subtest is a **diagnostic set for the emoji channel**, which is why the 2023 multimodal lift is larger there.

The full test set has **no** `<user>` mentions (0%) while training has 23.7%. That is a second distribution shift the original notebooks never named.

## Hashtags leak the label

`#not` appears in 3,188 training tweets and is sarcastic in **97.4%** of them. `#yeahright`, `#sarcastictweet`, and `#sarcastic` sit in the mid-to-high 90s. On the **test** set, 61.5% of sarcastic tweets carry a lexicon sarcasm hashtag; on the **subtest**, 77.9%. Almost none of the non-sarcastic test/subtest tweets do (0.0% in the cue table).

That is the distant-supervision trick from Davidov et al. (2010) sitting inside the evaluation sets. A one-line rule — “predict sarcastic if `#not` / `#sarcasm` / … is present” — already gets:

| Split | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| test | 81.30% | 99.53% | 62.90% | 77.08% |
| subtest | 88.85% | 100.00% | 81.98% | 90.10% |

So the 2023 Bi-LSTM numbers (87.35% / 89.21% multimodal accuracy) are partly “did the net see `#not`?”, not only “did emoji2vec help?”. Emoji still matter on the subtest (100% emoji, and multimodal RF jumps +4.7 accuracy), but hashtag leakage is the bigger story on these files.

## Bag-of-words without GloVe is already strong

On the same splits, `examples/lexical_baseline.py` (15k unigrams, no pretrained vectors):

| Model | test acc | subtest acc |
| --- | ---: | ---: |
| Multinomial NB | 84.75% | 85.61% |
| SGD logistic | 87.95% | 89.93% |
| Bi-LSTM + att. (2023, W) | 86.35% | 86.69% |
| Bi-LSTM + att. (2023, WE) | 87.35% | 89.21% |

SGD logistic matching the neural net is not a claim that LSTM is useless. It is a claim that **on this particular test set**, surface tokens (especially hashtags) already carry most of the label. A mean-pooled GloVe SVM at 76.9% is weaker than bag-of-words because averaging washes `#not` into a 200-D soup.

## Label noise

`#not` is not a perfect label. About 2.6% of training `#not` tweets are marked 0. One of the first training rows is:

```text
05 really taught me a valuable lesson I'm never gonna be late again! #Not   gold=0
```

The example tokenizer flags that hashtag; the gold label does not. Treat distant-supervision tags as a strong feature, not as ground truth.

## Tokenizer length vs Keras `maxlen`

This repo’s example tokenizer measures a training max of **90** tokens. The saved Keras models were built with `maxlen=78` from NLTK `TweetTokenizer`. Different splitters, different pad length. Do not mix the two embedding matrices.

## Errors the hashtag rule cannot see

`examples/error_analysis.py` lists sarcastic test tweets the rule misses: contrast without a tag (“So many useless classes , great to be student”, polite-thank-you that is actually a complaint). Those 371 false negatives are the part of the problem where a Bi-LSTM *could* still earn its keep. The three rule false positives are mostly ellipsis / “I love …” stems without a sarcasm tag.

Bag-of-words NB flips the other way: high recall (938/1000) and 243 false positives, several of them ordinary “speak to you” lines. Unigram “speak” is a bad sarcasm feature; the net’s attention over time is the intended fix, but only if the embedding table keeps `#not` and the emoji rows distinct.

## Token log-odds

`examples/token_odds.py` ranks `log P(w|sarcastic) − log P(w|not)` on the training set (`min_df=4`). The top of the sarcastic list is the distant-supervision tag set. That is independent confirmation of the cue table: the classifier that “learns” `#not` is recovering the labeling process.

## What the attention walk-through shows

`examples/attention_walkthrough.py` builds a toy sequence “I love walking to `<pad>` `<pad>` school `#not`”. With a weight vector that prefers the reversal dimensions, attention mass sits on the last two steps and the mask zeros the pads. Unmasking those pads steals ~0.17 total probability and moves the context vector (L2 ≈ 0.20). That is why `attention_layer.py` applies the mask **after** `exp` and why a per-timestep bias of length 78 is baked into the trained layer.
