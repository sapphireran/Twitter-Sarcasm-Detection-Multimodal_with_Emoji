# References

Personal reading list for this 2023 UCPH Computational Cognitive Science 2 project. Links are to the public papers and resources the code comments or methods depend on — not to any company-internal doc.

## Methods used in the code

- Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck. *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems.* arXiv:1512.08756. The scoring form implemented in `attention_layer.py` (`e = tanh(x W + b)`, masked softmax, weighted sum).
- Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, Sebastian Riedel. *emoji2vec: Learning Emoji Representations from their Description.* Workshop on NLP for Social Media, EMNLP 2016. arXiv:1609.08359. Source of `emoji2vec.bin` (300-d, Google-News space) and the idea behind `emoji2vec_twitter.bin` (200-d, Twitter GloVe space).
- Jeffrey Pennington, Richard Socher, Christopher D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014. [Project page](https://nlp.stanford.edu/projects/glove/) hosts `glove.twitter.27B.200d.txt`, which is **not** checked in here.
- Sepp Hochreiter, Jürgen Schmidhuber. *Long Short-Term Memory.* Neural Computation 9(8), 1997. The recurrent cell inside `Bidirectional(LSTM(256, …))`.
- Mike Schuster, Kuldip K. Paliwal. *Bidirectional Recurrent Neural Networks.* IEEE TSP 45(11), 1997.

## Sarcasm / irony on Twitter (context for the dataset)

The files in `dataset/` are a course snapshot. They show the usual distant-supervision fingerprints (`#not`, `#sarcasm`, `#sarcastictweet`, `#yeahright`). These papers are the standard context for that collection style; they are **not** a claim of exact corpus identity:

- Ellen Riloff, Ashequl Qadir, Prafulla Surve, Lalindra De Silva, Nathan Gilbert, Ruihong Huang. *Sarcasm as Contrast Between a Positive Sentiment and Negative Situation.* EMNLP 2013.
- Dmitry Davidov, Oren Tsur, Ari Rappoport. *Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon.* CoNLL 2010.
- Cynthia Van Hee, Els Lefever, Véronique Hoste. *SemEval-2018 Task 3: Irony Detection in English Tweets.* SemEval 2018.
- Aditya Joshi, Pushpak Bhattacharyya, Mark J. Carman. *Automatic Sarcasm Detection: A Survey.* ACM Computing Surveys 2017.

## Software

- TensorFlow / Keras 2 Sequential API — `dl_model.py`, SavedModel directories under `model/`.
- `keras_preprocessing` `Tokenizer` and `pad_sequences` — `data_utils.Preprocess`.
- NLTK `TweetTokenizer` — `data_utils.ReadOpen`.
- Gensim `KeyedVectors` — notebook loads of GloVe and emoji2vec.
- scikit-learn — SVM, trees, forests, GBT, and the metric functions.
- `emoji` (carpedm20) — `emoji_list` / `is_emoji` in the OOV fallback.

## Course framing

- University of Copenhagen, Computational Cognitive Science 2, 2023 final project.
- Working title, from the original README: *Multi-modal Sarcasm Detection Using Textual Contents and Emoji Co-occurrences in Twitter.*
- License: MIT, copyright 2023 `pang990801` (see `LICENSE`). This documentation expansion is for the personal GitHub snapshot `sapphireran/Twitter-Sarcasm-Detection-Multimodal_with_Emoji`.
