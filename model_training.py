import nltk
nltk.download('punkt')
nltk.download('stopwords')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Sample dataset
conversations = [
    ("hi", "hello"),
    ("how are you?", "I'm fine"),
    ("what's your name?", "I am a chatbot"),
    ("bye", "goodbye")
]

questions, answers = zip(*conversations)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(questions)
target_sequences = tokenizer.texts_to_sequences(answers)

max_len = 10  # fixed for both input/output
input_padded = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_padded = pad_sequences(target_sequences, maxlen=max_len, padding='post')
target_output = np.expand_dims(target_padded, -1)

# Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(input_padded, target_output, epochs=300, verbose=0)

# Save model & tokenizer
model.save("chatbot_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

