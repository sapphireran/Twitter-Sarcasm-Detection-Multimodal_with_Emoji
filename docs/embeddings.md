# Embeddings

Two vector tables sit in the repo. The third one the notebooks need is too
large for git.

| File | Tokens | Dim | In git? | Used for |
| --- | ---: | ---: | --- | --- |
| `glove.twitter.27B.200d.bin` / `glove_tt.txt` | 1.2M words | 200 | No | Every trained model |
| `emoji2vec_twitter.bin` | 1,661 | 200 | Yes | Multi-modal path (`WE`) |
| `emoji2vec.bin` | 1,661 | 300 | Yes | Alternate table; not wired into `PrepModel` |

The 200-d Twitter table is the one that matches GloVe. That is why
`AverageVectorPerEmoji` can concatenate onto a 200-d word mean and why
`Preprocess` can write an emoji2vec row into the same 200-d embedding
matrix. The 300-d file is the upstream emoji2vec release; leave it alone
unless you change the model width.

`examples/inspect_emoji2vec.py` only needs Gensim and the 200-d file:

```bash
python3 examples/inspect_emoji2vec.py --neighbors "😒" --k 8
python3 examples/inspect_emoji2vec.py --report
```

## What the neighbours look like

Nearest neighbours in `emoji2vec_twitter.bin` from a Gensim `most_similar`
call (not cherry-picked beyond the queries):

| Query | Neighbours (cosine) |
| --- | --- |
| 😒 | 😞 0.51, 🙎 0.49, 🙁 0.48, 😟 0.48, 😨 0.47 |
| 😂 | 😹 0.69, 😢 0.60, 😭 0.51, 😊 0.46 |
| 😍 | run `--neighbors` to refresh; faces cluster with other faces |

The geometry is coarse. 😂 sits next to 😹 and also 😢 / 😭, which is the
usual "this glyph is used in emotional tweets" collapse, not a clean
happy/sad axis. That is one reason a 400-d bag-of-means SVM can get *worse*
when you add emoji, while a Bi-LSTM that sees the glyph in sequence can
still pick up a clash with `I love…`.

## How the original code looks them up

`AverageVectorPerEmoji` only hits the table when the NLTK token *is* the
emoji. `Preprocess` is more aggressive: if a token misses GloVe it runs
`emoji.emoji_list` over the string and averages every `is_emoji` character.
A token like `:(` or a glued `love😭` can therefore receive an emoji2vec
row even though it is not a standalone glyph.

Gensim 4 renamed `KeyedVectors.vocab` to `key_to_index`. The 2023
`data_utils.py` still writes `if j in model_word2vec.vocab`. That works on
Gensim 3 and breaks on Gensim 4. The inspect script uses the Gensim 4 API.
If you rerun the notebooks on a current environment, change `.vocab` to
`.key_to_index`.

## Vocabulary coverage on this corpus

`examples/inspect_emoji2vec.py --report` walks every unique emoji extracted
from train/test/subtest and counts how many sit in the 200-d table:

| Split | Unique emoji in-table |
| --- | --- |
| train | 402 / 439 |
| test | 91 / 99 |
| subtest | 91 / 99 |

The common faces (😂 😒 😭 😍 😊) are present. Typical misses are older
miscellaneous symbols and text-style hearts (`☺ ♡ ♥ ☀ ☆`). Regional-indicator
flags written with spaces (`🇺 🇸`) and some circled glyphs (`⭕`) are the
same tokens that made two subtest tweets look "emoji-free" to the
lightweight extractor.
