# Walkthrough (measured on this branch)

These numbers come from `python3 -m examples.run_all` after the docs/examples
pass. Hash embeddings and the logistic seed are deterministic, so a re-run
should match.

## 1. The real CSV shift

`inspect_dataset.py` on the checked-in files:

| Split | n | sarcastic | emoji rate | `#not` / `#sarcasm` family |
| --- | ---: | ---: | ---: | ---: |
| train | 39,780 | 46.5% | 13.7% | 9.2% |
| test | 2,000 | 50.0% | 13.8% | 30.5% |
| subtest | 278 | 61.9% | **99.3%** | 47.8% |

The 2023 multimodal lift lives on the last row. The balanced test set is only
13.8% emoji, so concatenating a second 200-d channel is mostly concatenating
zeros. The subtest is almost all emoji.

Top hashtag on every split is `#not` (3,191 / 465 / 110).

## 2. Tokenize → average → concatenate

Tweet:

```
i just love getting shots 💉 #sarcastictweet
```

Tokens (same rules as `ReadOpen`: lowercase, keep hashtags and emoji):

```
['i', 'just', 'love', 'getting', 'shots', '💉', '#sarcastictweet']
```

Text-channel hits: the six non-emoji tokens.
Emoji-channel hits: `💉` only.
Shapes on the 16-tweet toy set: `X_text (16, 32)`, `X_multi (16, 64)`.

## 3. The illustrative pair

| Tweet | text-channel L2 | emoji-channel L2 |
| --- | ---: | ---: |
| `i love mondays 😒 #not` | 0.591 | 1.000 |
| `i love mondays` | 0.623 | **0.000** |

Cosine of the two text halves: **0.914** (same words; `#not` keeps it under 1).
Cosine of the two 64-d multimodal rows: **0.465**.

That drop is Fusion A. The second half is a unit emoji vector versus zeros,
which is exactly `AverageVectorPerEmoji` on a tweet with no pictograph.

## 4. Attention

Synthetic 8-step hidden state, spike at `t=5`, Raffel formula from
`attention_layer.py`:

```
t=0..4,6,7   α ≈ 0.070
t=5          α = 0.511   argmax
sum(α)       1.000
```

Mask step 5 as pad: `α_5 = 0`, remaining mass ~0.14 each. Flat hidden + no
bias: `α = (0.25, 0.25, 0.25, 0.25)`.

## 5. Toy logistic regression

16 tweets, 8/8, hash embeddings, numpy log-reg, seed 1.

| View | acc | precision | recall | F1 | final loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| text 32-d | 0.875 | 0.875 | 0.875 | 0.875 | 0.670 |
| multi 64-d | 0.875 | 1.000 | 0.750 | 0.857 | 0.615 |

Same accuracy, better loss, higher precision — the same *direction* as the
2023 multimodal Bi-LSTM on the real test set (precision 0.853 → 0.904, recall
0.879 → 0.836). Do not treat 0.875 as a project result; the 16-tweet set is
a teaching fixture.

`P(sarcastic)` on the illustrative pair:

| Tweet | text | multi |
| --- | ---: | ---: |
| `i love mondays 😒 #not` | 0.557 | 0.670 |
| `i love mondays` | 0.535 | 0.472 |
| gap | 0.022 | **0.198** |

The extra 32 dimensions are doing the work the GloVe average cannot.

## 6. Commands

```bash
python3 examples/inspect_dataset.py
python3 examples/run_pipeline.py
python3 examples/run_attention.py
python3 examples/run_toy_classifier.py
python3 examples/run_real_tokens.py
python3 -m examples.run_all
python3 -m unittest discover -s tests -v
```
