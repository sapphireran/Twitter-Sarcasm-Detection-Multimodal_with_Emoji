# Reproduction

Two different “re-run” levels exist.

## Level A — archive lab (this branch)

Needs Python 3.10+ and NumPy.

```bash
python3 -m pip install -r requirements-examples.txt
PYTHONPATH=. python3 -m ccs2lab --skip-baseline
PYTHONPATH=. python3 examples/06_hashed_baseline.py --train-limit 12000
PYTHONPATH=. python3 -m unittest discover -s tests
```

This path reads the CSVs and the two emoji2vec binaries. It does not
load Keras or sklearn pickles. It will not reproduce the 0.8735 LSTM
figure. It *will* reproduce the split census, cue-shift table, and
attention identities.

## Level B — 2023 notebooks

Needs the historical stack in `requirements.txt` plus files that are
not in git:

| Artifact | Status |
| --- | --- |
| `dataset/*.csv` | present |
| `emoji2vec_twitter.bin` | present (1,661 × 200) |
| `emoji2vec.bin` | present (1,661 × 300), unused |
| GloVe Twitter 27B 200d | **missing** |
| `model/best_model_*/saved_model.pb` | present |
| `model/best_model_*/variables/` | **missing** |
| DT / GBT pickles | present |
| SVM / RF pickles | **missing** |

`evaluate_loaded_dl_models.ipynb` calls
`tf.keras.models.load_model("model/best_model_multi_modal")`. A
SavedModel without `variables/` will load the graph and fail on
`evaluate`. The notebook cell outputs from 2023-06-06 are the only
remaining evidence of those weights.

To rebuild the LSTM you would need GloVe, `PrepModel`, and a training
loop that the repo never saved as a script (only implicit notebook
state). `dl_model.py` is the architecture; it is not a trainer.

## Suggested GloVe recovery (not done here)

The notebooks name two files:

* `glove.twitter.27B.200d.bin` (binary, `baseline_models.ipynb`)
* `glove_tt.txt` (text, `get_metrics_of_models.ipynb`)

Both are conversions of Stanford’s `glove.twitter.27B.200d.txt`.
Convert with gensim’s `glove2word2vec` / `KeyedVectors.save_word2vec_format`
if you have the official 1.4 GB Twitter GloVe dump. Do not commit it;
it is not this project’s data.

## Randomness

`ml_read_data` shuffles with an unseeded `np.random.permutation`.
Sklearn models used default seeds. LSTM training seeds are not
recorded. Exact pickle-level reproduction was already unlikely in
2023; it is impossible from this snapshot.
