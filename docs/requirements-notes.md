# Dependency notes

Two environments, intentionally separate.

## Examples and tests (this snapshot)

```
python >= 3.10
numpy
```

NumPy is already imported by the original `data_utils.py`. The
example CLIs and `tests/test_examples.py` use only the stdlib on top
of it. No `pip install` is required on a machine that already had
the course stack; on a clean box:

```bash
python3 -m pip install numpy
python3 -m unittest tests.test_examples
```

## 2023 notebooks (not pinned, not fully present)

See [reproduction.md](reproduction.md). The notebooks want TensorFlow
2 + Keras 2, gensim 3.x, NLTK, `emoji`, pandas, scikit-learn, joblib,
and matplotlib, plus the missing Twitter GloVe binary.

There was never an upstream `requirements.txt`. Do not treat a
modern `tensorflow>=2.16` / `gensim>=4` install as a drop-in: the
saved models use `tensorflow.python.keras` wrappers and the code
calls `KeyedVectors.vocab`.
