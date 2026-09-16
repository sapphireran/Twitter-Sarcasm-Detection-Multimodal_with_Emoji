# Reproduction

Two tracks: the **2023 notebooks** (TensorFlow, Gensim, NLTK, GloVe) and the **examples/** track (Python 3 + NumPy).

## Examples track (what this branch adds)

```bash
python3 -m pip install -r requirements-examples.txt   # numpy
python3 examples/dataset_overview.py
python3 examples/cue_analysis.py
python3 examples/lexical_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/tokenize_demo.py
python3 -m unittest discover -s tests -v
```

No GloVe, no emoji2vec load, no GPU. Scripts resolve files from the repo root via `examples/sarcasm_lab/paths.py`.

## 2023 notebook track

Approximate training environment from the notebooks (`language_info` still says Python 2.7.6 in one file; the execution timestamps and TensorFlow `evaluate` logs are Python 3 / TF 2):

| Package | Used for |
| --- | --- |
| `tensorflow` / `keras` | `PrepModel`, SavedModel load |
| `gensim` 3.x | `KeyedVectors.load_word2vec_format`, `.vocab` |
| `nltk` | `TweetTokenizer` |
| `emoji` | `emoji_list` / `is_emoji` inside `Preprocess` |
| `pandas`, `numpy` | labels and matrices |
| `scikit-learn`, `joblib` | baselines |
| `keras_preprocessing` | `Tokenizer`, `pad_sequences` |

### External files the notebooks open

| Path in notebook | In this repo? |
| --- | --- |
| `glove.twitter.27B.200d.bin` or `glove_tt.txt` | **No** — download [GloVe Twitter 27B](https://nlp.stanford.edu/projects/glove/) (200d) and convert if you need binary |
| `emoji2vec_twitter.bin` | Yes (repo root) |
| `emoji2vec.bin` | Yes (unused by the notebooks, kept as an alternate dump) |
| `dataset/*.csv` | Yes |
| `train_sentence.csv` next to the notebook | Only under `dataset/` — copy or fix paths |
| `baseline_models/svm_classifier.pkl` and `rf_*.pkl` | **No** |
| `baseline_models/dt_*.pkl`, `gbt_*.pkl` | Yes |
| `model/best_model_*` including `variables/` | Graphs only; **variable shards missing** |

### Gensim 4

Replace `token in model.vocab` with `token in model.key_to_index` and `model[token]` with `model.get_vector(token)` if you upgrade.

### Keras attention + custom objects

`tf.keras.models.load_model("model/best_model_multi_modal")` needs the `Attention` class registered (`custom_objects={"Attention": Attention}`) on modern Keras. The 2023 notebook did not pass that argument; it depended on the then-current saved-model wrappers.

`PrepModel` uses `Adam(lr=...)`. Current Keras wants `learning_rate=`.

### Do not retrain the “except FileNotFoundError” cells blindly

In `baseline_models.ipynb`, several fallback constructors instantiate `SVC()` while the section title says Decision Tree / Random Forest / Gradient Boosting. Prefer the recorded metrics over a fresh fit from those cells.

## Suggested layout if you restore the full stack

```text
.
├── dataset/                 # already here
├── glove.twitter.27B.200d.bin
├── emoji2vec_twitter.bin    # already here
├── baseline_models/         # restore svm / rf pickles if you have them
└── model/
    ├── best_model_single_modal/   # needs variables/
    └── best_model_multi_modal/
```

## Privacy

Tweets are mention-anonymized (`<user>`) but still user-generated text. Do not republish the CSVs as a new public dataset. The example scripts only print aggregate counts plus a handful of already-committed example lines.
