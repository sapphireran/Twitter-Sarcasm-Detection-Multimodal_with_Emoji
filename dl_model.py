"""Bi-LSTM + Raffel attention classifier used in the 2023 notebooks.

`PrepModel` is the graph only; there is no fit loop in this file.
Architecture notes: `docs/architecture.md`. A NumPy attention demo:
`examples/attention_walkthrough.py`.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Activation, Bidirectional, Dense, Flatten, Embedding
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import *
from tensorflow.keras.optimizers import Adam
from attention_layer import Attention


def PrepModel(count, embedding_matrix, l, lrate=0.001):
    model = Sequential()
    e = Embedding(count, 200, weights=[embedding_matrix], input_length=l, trainable=False)
    model.add(e)
    model.add(Dropout(0.25))
    model.add(Bidirectional(
        LSTM(256, kernel_initializer='he_normal', recurrent_activation='sigmoid', return_sequences=True,
             activation='tanh')))
    model.add(Dropout(0.4))
    model.add(Bidirectional(
        LSTM(256, kernel_initializer='he_normal', recurrent_activation='sigmoid', return_sequences=True,
             activation='tanh')))
    model.add(Dropout(0.4))
    model.add(Attention())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=['acc'])
    return model
