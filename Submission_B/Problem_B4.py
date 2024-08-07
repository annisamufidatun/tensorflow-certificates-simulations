# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

# buat callback untuk menghentikan training
class stopper(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.92 and logs.get('val_accuracy')>0.91):
            print("\nAkurasi di atas 91%. Hentikan training!!!")
            self.model.stop_training = True
def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    labels = bbc["category"].values.tolist()
    sentences = bbc["text"].values.tolist()

    training_size = int(len(sentences) * training_portion)

    training_sentences, val_sentences = sentences[:training_size], sentences[training_size:]
    training_labels, val_labels = labels[:training_size], labels[training_size:]

    #Untuk data sentences
    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    # Generate and pad the training sequences
    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded = pad_sequences(val_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    #Untuk data label
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    train_label = label_tokenizer.texts_to_sequences(training_labels)
    val_label = label_tokenizer.texts_to_sequences(val_labels)

    train_label = np.array(train_label).flatten()
    val_label = np.array(val_label).flatten()

    # You can also use Tokenizer to encode your label.

    callback = stopper()
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Last layer should not be changed
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(
        train_padded,
        train_label,
        epochs=100,
        validation_data=(val_padded, val_label),
        callbacks=[callback],
        verbose=2
    )
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
