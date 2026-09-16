# Limitations

A short list of things this personal 2023 project does **not** establish.

## Distant-supervision leakage

A large sarcastic subset is labeled in the style of `#not` / `#sarcasm` / `#sarcastictweet`. Those strings are still in the input. Every neural and classical number in [`models-and-results.md`](models-and-results.md) is therefore an upper bound on “sarcasm after the hashtag is gone.” The heuristic example makes the gap visible; the original experiments did not close it.

## Subtest is a probe, not a second test set

278 rows, 62% sarcastic, selected to be emoji-heavy. Confidence intervals are wide. A 4.7-point random-forest jump on that slice is suggestive, not a deployment metric.

## Frozen 2014-era embeddings

Twitter GloVe 27B 200-d and emoji2vec are static. They do not know post-2014 emoji, and they cannot move a word’s meaning when it is used sarcastically. The LSTM can only remix those frozen directions.

## Untuned classical models

`SVC()`, `RandomForestClassifier()`, and friends were used with library defaults. A weaker multimodal SVM on the test set may be a hyperparameter story, not a feature-fusion story.

## Incomplete artifacts

SVM/RF pickles and the Keras weight shards are not fully in git. The notebook outputs are the scientific record. You cannot, from this checkout alone, regenerate the 0.8735 multimodal accuracy.

## No significance tests, no error taxonomy

There is no bootstrap, no McNemar table, and no breakdown of errors into “missed polarity flip,” “literal positive,” “quoted lyrics,” “hashtag-only,” etc. Without that, it is easy to overfit a story to the aggregate table.

## Software rot

`dl_model.py` imports `tensorflow.python.keras` private modules and uses `Adam(lr=...)`. `data_utils.py` uses Gensim 3’s `.vocab`. `Attention` has no `get_config`. This is expected for a 2023 student repo and is the main reason the **examples** are stdlib-only.

## Not a social-media product

There is no live Twitter/X client, no scraper, and no authentication flow. Expanding this repo should stay on the local CSVs. If you want a modern sarcasm model, start from a current licensed dataset and a current encoder; do not treat these checkpoints as a shipped service.

## Personal scope

This work was a UCPH CCS2 final project. It is not affiliated with any employer. Do not copy these files into a company tree or present the numbers as an industry benchmark.
