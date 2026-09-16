# Example scripts

All commands assume the repository root and Python 3.10+. NumPy must be
importable. None of these scripts load GloVe, TensorFlow, or NLTK.

## Dataset snapshot

```bash
python3 -m examples.dataset_report
python3 -m examples.dataset_report --markdown -o docs/generated/dataset_snapshot.md
```

Prints split sizes, class balance, emoji / hashtag coverage, cue rates, and
train/test overlap. Use this when you want numbers that match the CSVs rather
than memory of the 2023 slides.

## Emoji log-odds

```bash
python3 -m examples.emoji_signals --split test --limit 12
python3 -m examples.emoji_signals --split train
```

Ranks emoji by smoothed log-odds of the sarcastic class. Test and train
disagree: 😂 is everywhere on train, while 😒 / 😑 / 🔫 mark sarcasm more
cleanly on the official test set.

## Tokenizer walkthrough

```bash
python3 -m examples.tweet_tokenizer_demo
```

Runs the example tokenizer on `examples/sample_tweets.json`. Shows the comma
rewrite from `ReadOpen`, kept hashtags, and the hand-built contrast features.
The Christmas `#not ready yet` tweet is included on purpose: `#not` is not
always a sarcasm label.

## Attention walkthrough

```bash
python3 -m examples.attention_demo
```

Ports `attention_layer.Attention` to NumPy and scores a 3-step sequence
`love → dirty-house → #not`. A scorer that likes the flip dimension puts most
of the softmax on `#not`. Mean-pooling cannot do that.

## Recorded 2023 metrics

```bash
python3 -m examples.metrics_table --metric accuracy
python3 -m examples.metrics_table --metric f1 --markdown
```

Pretty-prints the notebook numbers stored in
`examples/lib/recorded_metrics.py`. This is documentation, not a live
benchmark.

## Lexical logistic baseline

```bash
python3 -m examples.lexical_baseline_demo --train-limit 8000 --epochs 200
```

Trains a 19-feature logistic model on a random train subset and scores train
subset / test / subtest. Then it classifies the twelve hand-picked samples.

Expect `#not` and `contrast_cue` among the largest positive weights. Official
test accuracy will usually beat the train-subset accuracy because test is
denser in `#not` and `love … #not` templates.

## Tests

```bash
python3 -m unittest discover -s tests
```

Covers CSV loading, tokenizer cues, the NumPy attention identities, the
separable `#not` toy set, and the recorded-metrics table.

## Sample tweets

`examples/sample_tweets.json` is a personal mini-set, not an extra official
split. Each object has `id`, `text`, `label`, and a short `notes` field
explaining the cue.
