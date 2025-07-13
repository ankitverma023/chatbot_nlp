from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 10  # must match training value

# Predict reply
def predict_reply(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)
    pred_ids = np.argmax(pred[0], axis=1)
    reply_words = []
    for index in pred_ids:
        for word, i in tokenizer.word_index.items():
            if i == index:
                reply_words.append(word)
                break
    return " ".join(reply_words)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    reply = ""
    if request.method == "POST":
        user_input = request.form["message"]
        reply = predict_reply(user_input)
    return render_template("index.html", reply=reply)

if __name__ == "__main__":
    app.run(debug=True)
