# References

Only sources the code, notebooks, or embedding files actually depend on. This is not a literature review of sarcasm detection.

## Methods used in the repo

- Colin Raffel and Daniel P. W. Ellis. *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems.* arXiv:1512.08756, 2015. Implemented in [`attention_layer.py`](../attention_layer.py); the docstring cites this paper.
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014. Notebooks load **GloVe-Twitter 27B 200d**. Project page: <https://nlp.stanford.edu/projects/glove/>.
- Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. *emoji2vec: Learning Emoji Representations from their Description.* SocialNLP 2016. Files `emoji2vec.bin` and `emoji2vec_twitter.bin` in the repo root. Paper: <https://arxiv.org/abs/1609.08359>.

## Libraries the 2023 scripts import

- TensorFlow / Keras (`Sequential`, `Bidirectional`, `LSTM`, `Embedding`, custom `Layer`).
- `keras_preprocessing.text.Tokenizer` and `pad_sequences`.
- gensim `KeyedVectors.load_word2vec_format`.
- NLTK `TweetTokenizer`.
- `emoji` (`emoji_list`, `is_emoji`) for leftover tokens in `Preprocess`.
- scikit-learn: `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `accuracy_score`, `f1_score`, `precision_score`, `recall_score`.
- joblib for `baseline_models/*.pkl`.
- pandas / NumPy.

## Course framing

- University of Copenhagen, Computational Cognitive Science 2, 2023.
- Personal repository owner: Sapphire Ran (`sapphireran`), original commit author `pang990801`.
- License: MIT, 2023, see [`LICENSE`](../LICENSE).

## Related reading (not imported by the code)

These helped the 2023 write-up mentally but are **not** dependencies. Listed so a future personal note does not mistake them for something the notebooks fine-tuned.

- Ptáček, Habernal, Hong. *Sarcasm Detection on Czech and English Twitter.* COLING 2014. Classic hashtag-distant-supervision setup (`#sarcasm`).
- Rajadesingan, Zafarani, Liu. *Sarcasm Detection on Twitter: A Behavioral Modeling Approach.* WSDM 2015.
- Ghosh and Veale. *Fracking Sarcasm using Neural Network.* WASSA 2016. Early CNN/LSTM sarcasm work.

If you add a model, add a citation here only when the code or a notebook cell uses it.
