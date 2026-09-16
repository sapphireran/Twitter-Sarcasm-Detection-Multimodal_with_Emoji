"""Tweet loading, tokenization, and the two embedding views used in this project.

Personal CCS2 (2023) utilities. Nothing here is a general NLP library.

There are two consumers:

* Classical baselines (`ml_read_data`) get a mean-pooled GloVe vector per tweet
  and, for the WE setting, that vector concatenated with a mean-pooled
  emoji2vec vector. Order is thrown away on purpose.
* The BiLSTM (`Preprocess` / `preprocess_test`) keeps order. The multimodal
  trick is not a second tower: unknown emoji tokens are written into the
  *same* 200-d embedding table via emoji2vec when ``get_emoji2vec=True``.

Both paths lowercase with NLTK ``TweetTokenizer`` first. Sentence files are
not parsed as real CSV — see ``ReadOpen``.
"""

import emoji
import gensim
import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk import TweetTokenizer
from numpy import asarray, zeros


def ReadOpen(filename, Labelfile):
    """Read a parallel sentence / label pair the way the 2023 runs did.

    Sentences: raw lines, commas smashed to spaces, TweetTokenizer, lowercase.
    I did *not* use a CSV reader here. Quoted tweets with internal commas
    therefore tokenize differently than ``pandas.read_csv`` would.

    Labels: ``pandas.read_csv`` with no header. Expected values are 0 / 1.

    Returns
    -------
    data : list[list[str]]
        Tokenized tweets.
    labels : np.ndarray
        1-d label vector, same order as ``data``.
    count : int
        Number of sentence lines. ``Preprocess`` uses this as the embedding
        table *row* count, which is the number of documents, not vocab size.
        Wrong in theory; large enough in practice for this train split.
    """
    data = []
    tokenizer_tweet = TweetTokenizer()

    with open(filename, 'r', encoding="utf-8", errors="replace") as readFile:
        lines = readFile.readlines()

    for line in lines:
        temp = []
        # Comma-to-space is load-bearing. Do not "fix" this without retraining.
        sentence = ' '.join(line.strip().split(','))
        for token in tokenizer_tweet.tokenize(sentence):
            temp.append(token.lower())
        data.append(temp)

    labels_pd = pd.read_csv(Labelfile, index_col=False, header=None)
    labels = labels_pd.values.squeeze()

    return data, labels, len(lines)


def AverageVectorPerTweet(data, model_word2vec):
    """Mean-pool GloVe over in-vocabulary tokens. One 200-d row per tweet.

    Tokens missing from GloVe are skipped. A tweet with no hits becomes a
    zero vector — that does happen on very short or emoji-only lines.
    """
    avg = []
    for i in range(len(data)):
        row = []
        for j in data[i]:
            if j in model_word2vec.vocab:
                row.append(model_word2vec[j])
        if row:
            row = np.asarray(row)
            avg.append((np.average(row, axis=0)).tolist())
        else:
            avg.append(np.zeros((200,)).tolist())
    return avg


def AverageVectorPerEmoji(data, model_emoji2vec):
    """Mean-pool emoji2vec over in-vocabulary tokens. Same contract as GloVe.

    Most tweets have no emoji2vec hits, so most rows here are zeros. The
    WE baseline is then ``[glove; 0]``. sklearn can in principle learn to
    ignore that half; SVM on the full test in 2023 did not obviously do so.
    """
    avg = []
    for i in range(len(data)):
        row = []
        for j in data[i]:
            if j in model_emoji2vec.vocab:
                row.append(model_emoji2vec[j])
        if row:
            row = np.asarray(row)
            avg.append((np.average(row, axis=0)).tolist())
        else:
            avg.append(np.zeros((200,)).tolist())
    return avg


def ml_read_data(data_file, label_file, glove_model, emoji2vec_model):
    """Build the shuffled W and WE feature matrices for the sklearn baselines.

    W is 200-d mean GloVe. WE is 400-d (GloVe mean ∥ emoji2vec mean).
    One permutation is applied to both views so a row index still lines up.

    The shuffle is unseeded. Retraining will not match the 2023 pickles
    even if the CSV files are identical.
    """
    data, label, count = ReadOpen(data_file, label_file)

    embedded_sentences = AverageVectorPerTweet(data, glove_model)
    embedded_sentences_emoji = AverageVectorPerEmoji(data, emoji2vec_model)
    embedded_sentences_emoji = np.concatenate((np.array(embedded_sentences), np.array(embedded_sentences_emoji)),
                                              axis=1).tolist()
    X = np.array(embedded_sentences)

    X = np.array(X.tolist())
    y = np.array(label)
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    X_emoji = np.array(embedded_sentences_emoji)

    X_emoji = np.array(X_emoji.tolist())
    y_emoji = np.array(label)

    X_emoji = X_emoji[indices]
    y_emoji = y_emoji[indices]
    return X, y, X_emoji, y_emoji


def Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True):
    """Fit the train tokenizer and fill the frozen 200-d embedding table.

    ``get_emoji2vec=True`` is the WE setting: GloVe miss + peel-able emoji
    codepoints → mean emoji2vec row in the same slot. ``False`` writes
    zeros instead. Tokens GloVe already knows stay GloVe either way, so W
    is "do not impute emoji," not "delete emoji timesteps."

    ``count`` is ``len(train_lines)``. The matrix is therefore
    ``(n_docs, 200)`` rather than ``(vocab+1, 200)``. The 39,780-row train
    split is bigger than the Keras vocab, which is why this did not crash.

    Returns padded train sequences, the embedding matrix, pad length, and
    the fitted tokenizer (needed by ``preprocess_test``).
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, padding='post')
    maxlen = len(padded_docs[0])
    nf = 0
    embedding_matrix = zeros((count, 200))
    for word, i in tokenizer.word_index.items():
        if word in glove_model.vocab:
            embedding_matrix[i] = glove_model[word]
        else:
            new_em = []
            em = [item['emoji'] for item in emoji.emoji_list(word)]
            for ej in em:
                for c in ej:
                    if emoji.is_emoji(c):
                        new_em.append(c)
            try:
                if new_em:
                    row = []
                    for e in new_em:
                        row.append(emoji2vec_model[e])
                    if get_emoji2vec:
                        embedding_matrix[i] = np.average(np.asarray(row), axis=0).tolist()
                    else:
                        embedding_matrix[i] = [0] * 200
                else:
                    embedding_matrix[i] = [0] * 200
            except:
                # Broad on purpose in 2023: missing emoji2vec keys, empty
                # averages, the occasional weird codepoint. nf was never
                # printed in the surviving notebooks.
                embedding_matrix[i] = [0] * 200
                nf += 1

    return padded_docs, embedding_matrix, maxlen, tokenizer


def preprocess_test(tokenizer, maxlen, test_docs):
    """Encode test / subtest with the *train* tokenizer and pad length.

    Unknown tokens become 0 (Keras default) and therefore a zero embedding
    row. An emoji that never appeared in train cannot pick up emoji2vec
    at test time — the table was already frozen.
    """
    test_encoded_docs = tokenizer.texts_to_sequences(test_docs)
    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxlen, padding='post')
    return test_padded_docs
