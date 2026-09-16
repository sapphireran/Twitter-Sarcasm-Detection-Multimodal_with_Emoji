# References

## Papers the 2023 code actually implements

* Colin Raffel, Daniel P. W. Ellis. *Feed-Forward Networks with Attention Can
  Solve Some Long-Term Memory Problems.* 2016.
  <https://arxiv.org/abs/1512.08756>
  — scoring `e_t = tanh(h_t · W + b)` as used in `attention_layer.py`.

* Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, Sebastian
  Riedel. *emoji2vec: Learning Emoji Representations from their Description.*
  SemEval / *SocialNLP* 2016. <https://arxiv.org/abs/1609.08359>
  — source of `emoji2vec.bin` (300d) and the Twitter-aligned 200d table.

* Jeffrey Pennington, Richard Socher, Christopher D. Manning. *GloVe:
  Global Vectors for Word Representation.* EMNLP 2014.
  <https://nlp.stanford.edu/projects/glove/>
  — `glove.twitter.27B.200d`, not committed here.

## Background on Twitter sarcasm (not implemented, useful context)

* Dmitry Davidov, Oren Tsur, Ari Rappoport. *Semi-Supervised Recognition of
  Sarcastic Sentences in Twitter and Amazon.* CoNLL 2010. Distant
  supervision via `#sarcasm` / `#sarcastic`.

* Ellen Riloff, Ashequl Qadir, Prafulla Surve, Lalindra De Silva, Nathan
  Gilbert, Ruihong Huang. *Sarcasm as Contrast between a Positive Sentiment
  and Negative Situation.* EMNLP 2013.

* Meishan Zhang, Yue Zhang, Guohong Fu. *Tweet Sarcasm Detection Using Deep
  Neural Network.* COLING 2016. BiLSTM-style detectors on Twitter.

* Aniruddha Ghosh, Tony Veale. *Fracking Sarcasm using Neural Network.*
  WASSA 2016. CNN/LSTM sarcasm, emoji as features.

* Silvio Amir, Byron C. Wallace, Hao Lyu, Paula Carvalho, Mário J. Silva.
  *Modelling Context with User Embeddings for Sarcasm Detection in Social
  Media.* CoNLL 2016. User context; this repo does **not** use author ids.

## Software

* NLTK `TweetTokenizer` (`nltk.tokenize.casual`).
* Gensim `KeyedVectors.load_word2vec_format`.
* Keras 2 `Embedding` / `Bidirectional` / `LSTM` / `Dropout`.
* `emoji` PyPI package (`emoji_list`, `is_emoji`) in `data_utils.Preprocess`.

## Course

Computational Cognitive Science 2, University of Copenhagen, 2023. Personal
final project; not an official course solution set.
