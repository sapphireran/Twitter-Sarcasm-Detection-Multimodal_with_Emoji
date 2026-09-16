# Personal examples

Five scripts that run against the files already in this repository. They do
not download GloVe, do not load TensorFlow, and do not call a social-media
API.

Longer commentary: [`docs/examples.md`](../docs/examples.md).

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_preview.py
python3 examples/02_lexical_cues.py
python3 examples/03_emoji_vectors.py
python3 examples/04_attention_walkthrough.py
python3 examples/05_tfidf_baseline.py
python3 -m pytest tests/ -q
```

Or: `bash examples/run_all.sh`

| Script | Reads | Prints |
| --- | --- | --- |
| `01_dataset_preview.py` | `dataset/*.csv` | split sizes and labelled rows |
| `02_lexical_cues.py` | `dataset/*.csv` | `#not` / `#sarcasm` precision and a tag rule |
| `03_emoji_vectors.py` | `emoji2vec_twitter.bin` | neighbours of 😂 😒 😍 ❤️ |
| `04_attention_walkthrough.py` | nothing | softmax weights for a 4-step toy tweet |
| `05_tfidf_baseline.py` | `dataset/*.csv` | hashed TF-IDF accuracy, with and without hashtags |

Helpers used by the scripts and by `tests/` live in `examples/lib/`:

- `io.py` — headerless CSV pairing
- `tokenize.py` — regex tokenizer
- `word2vec_bin.py` — Gensim-free `.bin` reader
- `attention.py` — NumPy `tanh` attention
- `hashed_tfidf.py` — hash bag + logistic SGD
