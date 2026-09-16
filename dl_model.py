"""Keras constructor for the 2023 BiLSTM + attention sarcasm classifier.

Personal course code. ``PrepModel`` is the only public builder.

W vs WE is *not* decided here. Both settings use this graph. The difference
is how ``data_utils.Preprocess`` fills ``embedding_matrix`` before the
embedding layer is created with ``trainable=False``.

I never committed the ``fit`` loop. Optimizer / loss below are the only
training knobs that still live in git. Snapshot names in the notebooks
encode test acc and subtest acc, so I was watching both slices when I saved.

TF/Keras version note: ``Adam(lr=...)`` is the 2023 kwarg. Current Keras
wants ``learning_rate``. The mixed ``tensorflow.keras`` /
``tensorflow.python.keras`` imports are why reloaded graphs show
``module_wrapper_*`` layers.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Activation, Bidirectional, Dense, Flatten, Embedding
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import *
from tensorflow.keras.optimizers import Adam
from attention_layer import Attention


def PrepModel(count, embedding_matrix, l, lrate=0.001):
    """Frozen 200-d embedding → 2× BiLSTM-256 → attention → sigmoid.

    Parameters
    ----------
    count :
        First dimension of the embedding table. In the 2023 runs this was
        the number of *training documents* (see ``ReadOpen``), not vocab+1.
    embedding_matrix :
        ``(count, 200)`` initial weights. GloVe rows, optional emoji2vec
        imputations, otherwise zeros. Frozen after ``model.add``.
    l :
        ``input_length`` / pad length. 78 on the saved graphs. The
        attention bias is also length ``l``, so changing pad length makes
        old attention weights unloadable.
    lrate :
        Adam learning rate. Default 1e-3. I did not grid-search this.

    Dropout 0.25 after the embedding, 0.4 after each recurrent block.
    No recurrent dropout, no layer norm. He-normal on the LSTM input
    kernel; tanh activation / sigmoid recurrent activation.

    Output is one logit-less probability (sigmoid + binary cross-entropy).
    """
    model = Sequential()
    # trainable=False: 40k tweets should not wander 1.2M GloVe rows.
    # WE then has to win by having fewer zero rows at init, not by
    # leaving the emoji2vec neighborhood.
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
    # Sequences in, 512-d context vector out (2 × 256).
    model.add(Attention())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=['acc'])
    return model
