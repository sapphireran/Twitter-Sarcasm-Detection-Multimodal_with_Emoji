"""Tweet IO, mean-pooling, and the Keras embedding-matrix builder.

Coursework helpers used by the notebooks. A TensorFlow-free walkthrough
of the same ideas lives in `examples/` (see `examples/README.md` and
`docs/architecture.md`).
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
	data = []
	tokenizer_tweet = TweetTokenizer()

	with open(filename, 'r', encoding="utf-8", errors="replace") as readFile:
		lines = readFile.readlines()

	for line in lines:
		temp = []
		sentence = ' '.join(line.strip().split(','))
		for token in tokenizer_tweet.tokenize(sentence):
			temp.append(token.lower())
		data.append(temp)

	labels_pd = pd.read_csv(Labelfile, index_col=False, header=None)
	labels = labels_pd.values.squeeze()

	return data, labels, len(lines)


def AverageVectorPerTweet(data, model_word2vec):
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
                embedding_matrix[i] = [0] * 200
                nf += 1

    return padded_docs, embedding_matrix, maxlen, tokenizer


def preprocess_test(tokenizer, maxlen, test_docs):
    test_encoded_docs = tokenizer.texts_to_sequences(test_docs)
    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=maxlen, padding='post')
    return test_padded_docs
