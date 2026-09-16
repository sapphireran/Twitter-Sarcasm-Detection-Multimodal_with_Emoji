# Dataset

The committed files under `dataset/` are line-aligned tweet / label pairs. They are **not** RFC-4180 CSV: most tweets are a single field per line, some lines are wrapped in extra quotes, and commas inside a tweet are part of the text.

`data_utils.ReadOpen` therefore reads with `readlines()`, replaces commas with spaces, and tokenizes with NLTK `TweetTokenizer` (lowercased). Labels are a one-column integer file with no header.

## Splits

| Split | File pair | Rows | Label 0 (not sarcastic) | Label 1 (sarcastic) | Sarcasm rate |
| --- | --- | --- | --- | --- | --- |
| Train | `train_sentence.csv` / `train_label.csv` | 39,780 | 21,292 | 18,488 | 46.5% |
| Test | `test_sentence.csv` / `test_label.csv` | 2,000 | 1,000 | 1,000 | 50.0% |
| Subtest | `subtest_sentence.csv` / `subtest_label.csv` | 278 | 106 | 172 | 61.9% |

The test set is balanced. The subtest is smaller, sarcasm-heavy, and **every subtest tweet has at least one emoji** (`examples/dataset_overview.py`). Training and the full test set are only ~14% emoji. That is why every notebook reports **full test** and **subtest** separately: the subtest is a stress test for the emoji channel, not a second i.i.d. draw from the same distribution.

Two other shifts the 2023 notebooks did not name:

- The full test set has **0%** `<user>` mentions; training has 23.7%.
- Sarcasm hashtags (`#not`, `#sarcastictweet`, …) cover 61.5% of sarcastic **test** tweets and 77.9% of sarcastic **subtest** tweets, versus 19.7% of sarcastic **train** tweets. See [findings.md](findings.md).

## Text conventions

Tweets look like mid-2010s Twitter after mention anonymization:

- User mentions are already replaced with the token `<user>`.
- Hashtags are kept (`#not`, `#sarcastictweet`, `#Not`, `#yeahright`).
- Emoji appear as Unicode characters, often space-separated (`😭 😭 😭`).
- Some lines still contain leftover CSV quoting (`"100 words short of ..."`).

Typical sarcastic lines in the subtest lean on contrast between a positive stem and a cue:

```text
I loovee when people text back ... 😒 #sarcastictweet
I love watching golf ! ⛳ ️ #not
feeling like a million bucks after that chem 2 test . 😅 #not
```

Typical non-sarcastic training lines are ordinary status updates without those tags.

## What the two evaluation sets are for

- **Full test (2,000).** Head-to-head comparison of single-modal (word GloVe only) versus multimodal (word GloVe + emoji2vec). Both classical mean-pooled vectors and the Bi-LSTM see this split.
- **Subtest (278).** A slice where emoji and hashtag cues are unusually dense. Multimodal gains are larger here in the 2023 numbers (Bi-LSTM 86.69% → 89.21% accuracy). Treat it as a diagnostic set, not as the number to optimize.

## Paths used by the original notebooks

`baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb` open `train_sentence.csv` from the **working directory**, not `dataset/`. `get_metrics_of_models.ipynb` already uses `dataset/...`.

If you rerun the older notebooks, either copy the six files up one directory or edit the paths. The example scripts always read `dataset/` via `examples/sarcasm_lab/paths.py`.

## Provenance

This folder is the project-local extract used for the 2023 write-up. It is a Twitter sarcasm collection with mention anonymization; it is **not** a freshly scraped timeline. Do not treat the files as a license to redistribute raw Twitter content beyond this personal project.

Row counts above were computed from the committed files (one label per line). Re-run `python3 examples/dataset_overview.py` if the files change.
