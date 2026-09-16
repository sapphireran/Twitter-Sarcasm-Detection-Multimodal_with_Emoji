# Examples (stdlib walkthrough)

The 2023 notebooks need TensorFlow, Gensim, NLTK, and a 1.2 GB GloVe
file. The scripts under `examples/` do not. They only read `dataset/`
and, for the score table, `examples/reported_results.json`.

No script talks to Twitter/X or any other network service.

## Run

From the repository root:

```bash
python -m sarcasm_toolkit
python examples/01_inspect_dataset.py
python examples/02_tokenize_tweets.py
python examples/03_attention_pooling.py
python examples/04_lexicon_baseline.py
python examples/05_cue_logistic.py          # ~4k train rows
python examples/05_cue_logistic.py --full   # all 39,780 rows
python examples/06_pipeline_walkthrough.py
python -m unittest discover -s tests -v
```

Each example inserts the repo root on `sys.path` via `examples/_path.py`.

## What you should see

**01 inspect** — 39,780 / 2,000 / 278 rows, sarcastic rates 0.465 /
0.500 / 0.619, subtest emoji rate ~0.99, `#not` as the dominant tag.

**02 tokenize** — `#sarcastictweet` and `😒` stay whole tokens;
`<user>` stays one token; elongated `loovee` sets the elongation flag.

**03 attention** — with the toy `W` biased to cue/emoji axes, `#not`
and face emoji outrank `love` / `just`. Attention weights on one tweet
sum to ~1.

**04 lexicon** — high precision on every split; recall is modest on
train (~0.20 of sarcastic tweets have a cue tag) and much higher on
test/subtest (distant-supervision leak). Majority class is printed
beside it.

**05 logistic** — `any_cue_hashtag` / `has_not_hashtag` should dominate
the learned weights. Test accuracy will land near the lexicon, not near
the BiLSTM; that is expected.

**06 pipeline** — writes `examples/pipeline_last_run.json` (gitignored)
with the metrics dict from that run.

## Files

| Path | Role |
| --- | --- |
| `examples/sample_tweets.csv` | 10 hand-picked patterns |
| `examples/reported_results.json` | 2023 notebook scores |
| `sarcasm_toolkit/` | dataset IO, tokenizer, cues, attention, metrics |

Module-level notes: [modules.md](modules.md).
Command cheat-sheet: [../examples/README.md](../examples/README.md).
