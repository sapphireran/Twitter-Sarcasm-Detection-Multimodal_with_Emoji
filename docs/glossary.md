# Glossary

Terms as they are used in this personal project, not as a general NLP
dictionary.

**Single-modal / word-only / `_w`.**
A tweet represented only with GloVe Twitter word vectors. Emoji tokens
that are absent from GloVe become skipped (classical mean) or rows of
zeros (Keras matrix when `get_emoji2vec=False`).

**Multi-modal / word+emoji / `_we`.**
The same tweet with an emoji signal added. Classical models concatenate a
200-d emoji2vec mean. The LSTM writes emoji2vec rows into the **same**
200-d embedding table.

**Subtest.**
The 278 test tweets that contain at least one emoji. Not a third
independent sample from Twitter; it is a filter on the official test set.

**Cue hashtag.**
Tokens such as `#not`, `#sarcasm`, `#yeahright`, `#sarcastictweet`. They
are extremely predictive. A model that only looks at these markers can
look strong and still fail on unmarked sarcasm.

**GloVe Twitter 27B 200-d.**
Pennington et al. vectors trained on tweets. Dimension 200 is the table
the 2023 notebooks load. Not stored in this git repository.

**emoji2vec.**
Eisner et al. skip-gram vectors for emoji, trained so that emoji lie near
the words they co-occur with. `emoji2vec_twitter.bin` is the table the
notebooks load. `emoji2vec.bin` is the generic release sitting beside it.

**Mean pooling.**
Average of the in-vocabulary token vectors. Order disappears. A 40-token
sincere setup plus a final `#not` is diluted by the setup.

**Raffel attention.**
A learned scalar score per timestep, softmax across time, then a weighted
sum of hidden states. Implemented in `attention_layer.py` and mirrored in
`examples/attention_numpy.py`.

**Early fusion.**
Combine modalities before the classifier. Both 2023 stacks are early
fusion. There is no product-of-experts or gated late fusion.

**`<user>`.**
Anonymized mention. Counted as a mention cue by the lexical example
features.

**Positive class.**
Label `1` = sarcastic. F1 / precision / recall in the notebooks are the
binary (non-macro) scores for this class.
