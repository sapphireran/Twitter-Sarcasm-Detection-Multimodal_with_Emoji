# Multimodal Twitter Sarcasm Detection (Text + Emoji)

Personal 2023 University of Copenhagen project for **Computational Cognitive Science 2**. The work asks whether emoji co-occurrence adds anything useful on top of tweet text when the task is binary sarcasm detection.

This repository is personal academic work. It is not affiliated with an employer.

The original course upload was mostly notebooks plus saved models. This branch adds a written project record and examples that run against the shipped CSV splits without the large GloVe Twitter dump.

## What the project did

Sarcasm on Twitter is often a mismatch between the literal words and an extra signal: a hashtag (`#not`, `#sarcasm`), an emoji, or both. The project compared two embedding setups:

| Setup | Word stream | Extra signal |
| --- | --- | --- |
| Single-modal | GloVe Twitter 27B, 200-d | none |
| Multi-modal | same GloVe vectors | emoji2vec (200-d) when a token is an emoji or contains one |

Classical baselines (SVM, decision tree, random forest, gradient boosting) averaged those vectors per tweet. The neural model kept token order: frozen embedding → dropout → stacked bidirectional LSTMs → Bahdanau-style attention (Raffel et al., 2015) → sigmoid.

On the held-out test set the attention model reached **86.4%** accuracy with text only and **87.4%** when emoji vectors were mixed in. The gap is larger on the emoji-heavy subtest (**86.7%** vs **89.2%**). Random forest was the strongest classical baseline.

Full tables live in [`docs/baselines-and-results.md`](docs/baselines-and-results.md).

## Repository layout

```
attention_layer.py          Keras attention used by the BiLSTM
dl_model.py                 Sequential BiLSTM + attention builder
data_utils.py               tweet loading, averaging, Keras padding
baseline_models.ipynb       SVM / DT / RF / GBT training notes
evaluate_loaded_dl_models.ipynb
get_metrics_of_models.ipynb accuracy / F1 / precision / recall
dataset/                    train / test / subtest CSV pairs
emoji2vec.bin               original 300-d emoji2vec (1,661 tokens)
emoji2vec_twitter.bin       200-d emoji2vec aligned to GloVe Twitter
model/                      saved single- and multi-modal Keras graphs
docs/                       project write-up
examples/                   runnable walkthroughs on the shipped CSVs
tests/                      checks for the example library and data files
```

The original Keras SavedModel folders under `model/` only include `keras_metadata.pb` in this checkout. Reloading the 2023 weights needs the rest of those directories plus the GloVe binary, which was never committed (it is about 1.2 GB).

## Dataset in one screen

| Split | Tweets | Sarcastic | With emoji | Why it exists |
| --- | ---: | ---: | ---: | --- |
| `dataset/train_*.csv` | 39,780 | 46.5% | 13.8% | model fitting |
| `dataset/test_*.csv` | 2,000 | 50.0% | 13.9% | main held-out numbers |
| `dataset/subtest_*.csv` | 278 | 61.9% | 99.6% | emoji-present slice |

Labels are `1` = sarcastic and `0` = not sarcastic. The subtest is almost entirely tweets that contain at least one emoji, which is why the multi-modal gap shows up there more clearly than on the balanced test set.

See [`docs/dataset.md`](docs/dataset.md) for tokenisation, hashtag leakage, and label examples.

## Quick start (examples, no GloVe)

The example scripts use the Python standard library plus NumPy. They do not load TensorFlow or the Twitter GloVe file.

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_preview.py
python3 examples/02_lexical_cues.py
python3 examples/03_emoji_vectors.py
python3 examples/04_attention_walkthrough.py
python3 examples/05_tfidf_baseline.py
python3 -m pytest tests/ -q
```

What each script is for:

1. **Dataset preview** — split sizes, class balance, a few labelled rows.
2. **Lexical cues** — `#not` / `#sarcasm` / `#yeahright` versus emoji presence.
3. **Emoji vectors** — read `emoji2vec_twitter.bin` without Gensim and compare nearest neighbours.
4. **Attention walkthrough** — the same `tanh(xW + b)` pooling as `attention_layer.py`, in NumPy.
5. **TF-IDF baseline** — a from-scratch unigram classifier so there is a number you can reproduce on a laptop.

Walkthrough notes: [`docs/examples.md`](docs/examples.md) and [`examples/README.md`](examples/README.md).

## Recreating the 2023 neural numbers

That path still needs the missing GloVe Twitter 200-d vectors and a TensorFlow install that can load the original attention layer.

1. Install [`requirements.txt`](requirements.txt).
2. Place `glove.twitter.27B.200d.bin` (or the `glove_tt.txt` text dump used in `get_metrics_of_models.ipynb`) next to the notebooks.
3. Point the notebooks at `dataset/*.csv` (the metrics notebook already does; `baseline_models.ipynb` still uses bare filenames).
4. Load `model/best_model_single_modal` and `model/best_model_multi_modal` if the full SavedModel trees are present.

Details and known path mismatches: [`docs/reproduction.md`](docs/reproduction.md). Architecture notes: [`docs/architecture.md`](docs/architecture.md). Attention math: [`docs/attention.md`](docs/attention.md).

## Reported test-set accuracy

| Model | Text only | Text + emoji |
| --- | ---: | ---: |
| SVM | 0.769 | 0.763 |
| Decision tree | 0.727 | 0.730 |
| Random forest | 0.815 | 0.818 |
| Gradient boosting | 0.746 | 0.748 |
| BiLSTM + attention | **0.864** | **0.874** |

These figures are copied from the executed cells in `get_metrics_of_models.ipynb` and `baseline_models.ipynb`. They are historical course results, not numbers regenerated in this documentation pass.

A hashed TF-IDF logistic model that *does* run from this checkout scores **0.834** test accuracy with hashtags and **0.739** after they are dropped (`examples/05_tfidf_baseline.py`, seed 0). That gap is the label leak discussed in [`docs/dataset.md`](docs/dataset.md).

## Course and licence

- Course: Computational Cognitive Science 2, University of Copenhagen, 2023
- Author: Sapphire Ran (`sapphireran` / `pang990801`)
- Licence: MIT (see [`LICENSE`](LICENSE))

## References used in the original write-up

- Pennington, Socher, and Manning. *GloVe: Global Vectors for Word Representation*. EMNLP 2014. Twitter 27B 200-d vectors.
- Eisner, Rocktäschel, Augenstein, Bošnjak, and Riedel. *emoji2vec: Learning Emoji Representations from their Description*. SocialNLP 2016.
- Raffel and Ellis. *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems*. arXiv:1512.08756, 2015.
- Rajadesingan, Zafarani, and Liu. *Sarcasm Detection on Twitter: A Behavioral Modeling Approach*. WSDM 2015 (task framing).
