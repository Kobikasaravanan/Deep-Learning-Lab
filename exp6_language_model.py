# Language Modeling using Stacked Bidirectional RNNs
# Graph + Table Output

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# -----------------------------
# Step 1: Dataset
# -----------------------------

corpus = [
    "artificial intelligence is transforming the world",
    "machine learning is a part of artificial intelligence",
    "deep learning models are powerful",
    "natural language processing is interesting",
    "data science helps businesses make decisions",
    "neural networks learn patterns in data",
    "technology is evolving every day",
    "artificial intelligence is the future",
    "machine learning algorithms learn from data",
    "data is the new oil"
]

# -----------------------------
# Step 2: Tokenization
# -----------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

# -----------------------------
# Step 3: Input Sequences
# -----------------------------

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# -----------------------------
# Step 4: Padding
# -----------------------------

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = pad_sequences(input_sequences,
                                maxlen=max_sequence_len,
                                padding='pre')

input_sequences = np.array(input_sequences)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# -----------------------------
# Step 5: Build Model
# -----------------------------

model = Sequential()

model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# Step 6: Train Model
# -----------------------------

history = model.fit(X, y, epochs=50, verbose=1)

# -----------------------------
# Step 7: Create Table Output
# -----------------------------

results = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss'])+1),
    "Loss": history.history['loss'],
    "Accuracy": history.history['accuracy']
})

print("\nTraining Results Table:\n")
print(results)

# -----------------------------
# Step 8: Plot Graphs
# -----------------------------

plt.figure()

plt.plot(results["Epoch"], results["Loss"], label="Loss")
plt.plot(results["Epoch"], results["Accuracy"], label="Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Performance")
plt.legend()

plt.show()

# -----------------------------
# Step 9: Predict Next Words
# -----------------------------

def predict_next_words(seed_text, next_words):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list],
                                   maxlen=max_sequence_len-1,
                                   padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# -----------------------------
# Step 10: Test Prediction
# -----------------------------

print("\nPredicted Sentence:")
print(predict_next_words("artificial intelligence", 3))
