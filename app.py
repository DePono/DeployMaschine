import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import model_from_json
import streamlit
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The step after importing the required libraries is to load the tokenizer and deep learning model.
with open('tokenizer.pickle', 'rb') as tk:
    tokenizer = pickle.load(tk)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
lstm_model.load_weights("model.h5")


def target_prediction(message):
    input_message = [message]
    input_message = [x.lower() for x in input_message]
    input_message = [re.sub('[^a-zA-z0-9\s]', '', x) for x in input_message]
    input_feature = tokenizer.texts_to_sequences(input_message)
    input_feature = pad_sequences(input_feature, 145, padding='pre')
    target = lstm_model.predict(input_feature)[0]
    if np.argmax(target) == 0:
        pred = "Not spam"
    else:
        pred = "Spam"
    return pred


def run():
    streamlit.title("Sentiment Analysis - LSTM Model")
    html_temp = """
    """
    streamlit.markdown(html_temp)
    message = streamlit.text_input("Enter the message ")
    prediction = ""
    if streamlit.button("Predict target"):
        prediction = target_prediction(message)
    streamlit.success("The target predicted by Model : {}".format(prediction))


if __name__ == '__main__':
    run()
