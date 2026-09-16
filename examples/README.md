# Examples

Runnable walkthroughs for the personal sarcasm-detection repo. They use
`examples/sarcasm_lib/` (stdlib + NumPy) and the CSV files under `dataset/`.
They do **not** load GloVe, Keras checkpoints, or NLTK.

```bash
# from the repository root
python3 examples/inspect_dataset.py
python3 examples/emoji_signals.py --top 12
python3 examples/lexical_baseline.py
python3 examples/lexical_baseline.py --strip-supervision-tags
python3 examples/attention_demo.py
python3 examples/preprocess_walkthrough.py --prefer-emoji
python3 -m unittest discover -s tests -v
```

| Script | What it shows |
| --- | --- |
| `inspect_dataset.py` | Split sizes, class balance, emoji / hashtag / `#not` rates, short examples |
| `emoji_signals.py` | Monroe log-odds ranking of emoji and hashtags vs. the sarcastic class |
| `lexical_baseline.py` | From-scratch multinomial Naive Bayes vs. a pure `#not`/`#sarcasm` rule |
| `attention_demo.py` | NumPy port of `attention_layer.Attention` on a 2-row synthetic batch |
| `preprocess_walkthrough.py` | File line → ReadOpen rewrite → tokens → which embedding table would fire |

## Why the lexical baseline is leaky on purpose

A large share of class-1 labels are hashtag-supervised. Predicting “sarcastic”
exactly when `#not`, `#sarcasm`, `#yeahright`, or `#sarcastictweet` is present
is already a strong rule on this test file. The NB model will discover those
tags unless you pass `--strip-supervision-tags`. Use both numbers when you
write about “how much of the task is just reading the tag.”

Recorded in this checkout: tags-kept NB test acc **0.833**, tags-stripped
**0.744**, `#not`/`#sarcasm` rule **0.807**. Full table:
[docs/lexical_baseline.md](../docs/lexical_baseline.md).

## Library map

| Module | Responsibility |
| --- | --- |
| `sarcasm_lib/dataset.py` | Line-oriented loaders; optional ReadOpen-faithful comma rewrite |
| `sarcasm_lib/tokenize.py` | Hashtag / mention / emoji-aware tokenizer |
| `sarcasm_lib/features.py` | Cue flags, tag stripping, informative log-odds |
| `sarcasm_lib/naive_bayes.py` | Laplace-smoothed multinomial NB |
| `sarcasm_lib/attention.py` | Masked temporal attention |
| `sarcasm_lib/metrics.py` | Accuracy / precision / recall / F1 |

The original 2023 modules stay untouched so a future TensorFlow environment
can still import `data_utils` / `dl_model` as written.
