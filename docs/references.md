# References

Personal reading list for this course project. Links are to the public papers / vector releases, not to any private or employer document.

## Model pieces used in the repo

- Colin Raffel and Daniel P. W. Ellis. *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems.* 2015. [arXiv:1512.08756](https://arxiv.org/abs/1512.08756)  
  The attention equations in `attention_layer.py` (`e = tanh(x W + b)`, softmax over time, weighted sum).

- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014. [Paper](https://nlp.stanford.edu/pubs/glove.pdf) · [Twitter 27B download](https://nlp.stanford.edu/projects/glove/)  
  200-dimensional Twitter GloVe is the word channel. The binary dump is **not** committed.

- Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. *emoji2vec: Learning Emoji Representations from their Descriptions.* SocialNLP 2016. [arXiv:1609.08359](https://arxiv.org/abs/1609.08359)  
  Source of `emoji2vec.bin` / `emoji2vec_twitter.bin`.

- Sepp Hochreiter and Jürgen Schmidhuber. *Long Short-Term Memory.* Neural Computation 9(8), 1997.  
  LSTM cells inside `tf.keras.layers.LSTM`.

- Mike Schuster and Kuldip K. Paliwal. *Bidirectional Recurrent Neural Networks.* IEEE TSP 1997.  
  The `Bidirectional` wrapper around each LSTM.

## Sarcasm / social-text background

- Dmitry Davidov, Oren Tsur, and Ari Rappoport. *Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon.* CoNLL 2010.  
  Early hashtag-as-label work (`#sarcasm`, `#not`) that motivates the cue analysis in `examples/cue_analysis.py`.

- Ellen Riloff, Ashequl Qadir, Prafulla Surve, Lalindra De Silva, Nathan Gilbert, and Ruihong Huang. *Sarcasm as Contrast between a Positive Sentiment and Negative Situation.* EMNLP 2013.  
  The “love walking to school 😄 #not” pattern in the subtest.

- Aniruddha Ghosh and Tony Veale. *Fracking Sarcasm using Neural Network.* WASSA 2016.  
  CNN / LSTM sarcasm detectors that this Bi-LSTM + attention setup sits next to.

- Soujanya Poria, Erik Cambria, Devamanyu Hazarika, and Prateek Vij. *A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks.* COLING 2016.

- Aditya Joshi, Pushpak Bhattacharyya, and Mark J. Carman. *Automatic Sarcasm Detection: A Survey.* ACM Computing Surveys 2017.  
  Survey of cues, datasets, and why emoji / hashtags leak the label.

## Tokenization and tooling

- NLTK `TweetTokenizer` — the original `ReadOpen` splitter.  
  [NLTK tokenize API](https://www.nltk.org/api/nltk.tokenize.casual.html)

- `emoji` Python package — used only inside `Preprocess` to expand unknown tokens into emoji code points.

- TensorFlow Keras `Embedding` / `Bidirectional` / `LSTM` — `dl_model.py`.

## Course context

University of Copenhagen, Computational Cognitive Science 2, spring 2023 final project. Title as committed: *Multi-modal Sarcasm Detection Using Textual Contents and Emoji Co-occurrences in Twitter.*
