# Original file map

How the 2023 source files relate to the new `docs/` / `examples/` tree.

| File | What it does | Counterpart in this branch |
| --- | --- | --- |
| `data_utils.ReadOpen` | NLTK tweet tokenize + label load | `examples/sarcasm_lab/io.py`, `tokenize.py` |
| `data_utils.AverageVectorPerTweet` | mean GloVe | not reimplemented (needs GloVe) |
| `data_utils.AverageVectorPerEmoji` | mean emoji2vec | not reimplemented; cue counts stand in |
| `data_utils.ml_read_data` | 200-D / 400-D sklearn matrices | `examples/lexical_baseline.py` uses bag-of-words instead |
| `data_utils.Preprocess` | Keras ids + embedding table | documented in [preprocessing.md](preprocessing.md) |
| `attention_layer.Attention` | Raffel attention | `examples/sarcasm_lab/attention.py` |
| `dl_model.PrepModel` | Bi-LSTM + attention Sequential | [architecture.md](architecture.md) |
| `baseline_models.ipynb` | SVM / DT / RF / GBT | metrics copied into [results.md](results.md) |
| `evaluate_loaded_dl_models.ipynb` | Keras `evaluate` | same |
| `get_metrics_of_models.ipynb` | accuracy / F1 / P / R dump | same |
| `dataset/*.csv` | committed splits | `examples/dataset_overview.py` |
| `emoji2vec*.bin` | pretrained emoji vectors | unused by examples; still required by notebooks |
| `model/best_model_*` | SavedModel graphs | incomplete without `variables/` |
| `baseline_models/*.pkl` | some sklearn pickles | DT and GBT only |

## Call graph (2023 training)

```text
ReadOpen(train_sentence, train_label)
        │
        ├─► ml_read_data ─► mean GloVe / emoji2vec ─► sklearn.fit
        │
        └─► Preprocess ─► padded ids + embedding_matrix
                              │
                              └─► PrepModel ─► Sequential.fit
                                      │
                                      └─► Attention.call
```

## Call graph (examples)

```text
load_split ─► tokenize_tweet ─► extract_cues ─► cue rules
                     │
                     └─► CountVectorizer ─► MultinomialNB / SGDLogisticRegression
```
