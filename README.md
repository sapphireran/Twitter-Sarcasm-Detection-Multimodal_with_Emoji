# Multi-modal sarcasm detection (UCPH CCS2 2023)

Personal course archive for Computational Cognitive Science 2 at the
University of Copenhagen (2023). The project compared **word-only (W)**
and **word + emoji (WE)** sarcasm detectors on Twitter-style posts.

This branch adds a **stdlib + NumPy archive lab**. It does not retrain
the 2023 TensorFlow models and it does not contain company code.

## What the 2023 project did

| Track | Features | Models |
| --- | --- | --- |
| W | mean GloVe Twitter 27B 200d over tokens | SVM, DT, RF, GBT, BiLSTM + Raffel attention |
| WE | W vector concatenated with mean emoji2vec (200d) | same heads |

The deep model is a frozen 200d embedding, dropout, two bidirectional
LSTMs (256 units each), Raffel attention, and a sigmoid. Recorded
notebook scores:

| Model | Test W acc | Test WE acc | Subtest W acc | Subtest WE acc |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.769 | 0.763 | 0.813 | 0.824 |
| Decision tree | 0.727 | 0.730 | 0.777 | 0.799 |
| Random forest | 0.815 | 0.818 | 0.806 | 0.853 |
| Gradient boosting | 0.746 | 0.748 | 0.795 | 0.795 |
| BiLSTM + attention | 0.8635 | **0.8735** | 0.8669 | **0.8921** |

The WE gain is small on the official test set and larger on **subtest**.
That is not an independent third split. Subtest is the 278-tweet
emoji-bearing slice of test (100% overlap with test, 0 overlap with
train, 99.3% of rows contain an emoji). Test is also cue-heavier than
train: `#not` covers about 26% of test vs 9% of train.

## Archive lab (this branch)

Python 3.10+ and NumPy. The original TensorFlow / gensim / NLTK stack
is **not** required.

```bash
python3 -m pip install -r requirements-examples.txt
PYTHONPATH=. python3 -m ccs2lab --skip-baseline
PYTHONPATH=. python3 examples/06_hashed_baseline.py --train-limit 12000
PYTHONPATH=. python3 -m unittest discover -s tests
```

| Script | What it shows |
| --- | --- |
| `examples/01_split_census.py` | sizes, labels, overlaps, token lengths |
| `examples/02_cue_shift.py` | `#not` / `#sarcasm` coverage and Wilson intervals |
| `examples/03_emoji2vec_probe.py` | reads the checked-in 200d / 300d emoji tables |
| `examples/04_attention_replay.py` | NumPy Raffel layer, including the timestep bias |
| `examples/05_cue_rule.py` | hashtag-only baseline vs the 2023 SVM numbers |
| `examples/06_hashed_baseline.py` | hashed logistic + slice metrics |
| `examples/07_reprint_scores.py` | full recorded notebook table |

Notes live in [`docs/`](docs/README.md).

## What is missing from a full notebook re-run

* Stanford **GloVe Twitter 27B 200d** (`glove.twitter.27B.200d.bin` /
  `glove_tt.txt`) was never uploaded.
* `model/best_model_*/` SavedModels have `saved_model.pb` but no
  `variables/` shards.
* SVM and Random Forest pickles are absent; DT / GBT pickles are
  present. Some training cells in `baseline_models.ipynb` also
  instantiate the wrong sklearn class (SVC in the RF / GBT / DT-WE
  branches) if the pickle load fails.

## Layout

```
attention_layer.py     Raffel attention (Keras)
data_utils.py          tweet load + GloVe / emoji2vec averaging
dl_model.py            BiLSTM + attention builder
dataset/               train 39780 / test 2000 / subtest 278
emoji2vec*.bin         1,661 emoji vectors (300d and 200d)
ccs2lab/               archive lab library
examples/              runnable walkthroughs
docs/                  course-archive notes
```

MIT License (see `LICENSE`). Original upload: pang990801 / Sapphire Ran.
