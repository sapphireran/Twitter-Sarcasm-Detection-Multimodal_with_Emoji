# Project notes

These pages were written from the code, CSVs, and notebook outputs that are already in this personal repository. They do not add a new trained model. They exist so the 2023 course project can be read without opening three large notebooks and a pair of opaque `.bin` files.

| Page | Contents |
| --- | --- |
| [dataset.md](dataset.md) | Split sizes, label balance, emoji rate, sarcasm-cue hashtags, file format |
| [preprocessing.md](preprocessing.md) | `ReadOpen`, mean-pooling, embedding-matrix construction |
| [architecture.md](architecture.md) | Attention layer, BiLSTM stack, sklearn baselines |
| [experiments.md](experiments.md) | Recorded metrics and what they do / do not show |
| [reproducing.md](reproducing.md) | How to rerun notebooks vs how to rerun `examples/` |
| [notebooks.md](notebooks.md) | Cell-level map of the three original notebooks |
| [references.md](references.md) | Papers and embedding sources cited by the code |
| [results/recorded_metrics.json](results/recorded_metrics.json) | Machine-readable copy of the notebook numbers |

Related runnable scripts are listed in [`../examples/README.md`](../examples/README.md).

## How these notes were sourced

- Row counts and cue statistics: direct reads of `dataset/*.csv` (see `examples/01_dataset_overview.py`).
- Architecture: `dl_model.py`, `attention_layer.py`, `data_utils.py`.
- Metrics: printed cells in `get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb`.
- Saved artifacts: `model/best_model_*` Keras directories and `baseline_models/*.pkl`.

If a sentence here disagrees with a notebook cell, treat the notebook output as the 2023 experimental record and open an issue on this personal repo.
