import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import utils
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import re
import pickle

import re
import pickle

# load dataset from csv file
df_mails = pd.read_csv("combined_data.csv")
# print first 5 records
print(df_mails.head())
# print concise summary about dataset.
print(df_mails.info())
# Next, we apply bit of text preprocessing to clean the reviews using regular expressions.
df_mails['text'] = df_mails['text'].apply(lambda x: x.lower())
df_mails['text'] = df_mails['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# Checking for null values
df_mails.isna().sum()

# print no of duplicate records
print("Total duplicated records in dataset are : {}".format(df_mails.duplicated().sum()))

# lets remove duplicated records
df_mails.drop_duplicates(inplace=True)

df_mails = df_mails[:500].copy()

max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df_mails['text'].values)
X = tokenizer.texts_to_sequences(df_mails['text'].values)
X = pad_sequences(X)
print(X.shape)

embed_dim = 500
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(LSTM(10))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

y = pd.get_dummies(df_mails['label']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model.fit(X_train, y_train, epochs=5, verbose=1)

test = ['ounce feather bowl hummingbird opec moment alabaster valkyrie dyad bread flack desperate iambic hadron heft '
        'quell yoghurt bunkmate divert afterimage']
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, maxlen=X.shape[1], dtype='int32', value=0)
print(test.shape)
target = model.predict(test)[0]
if np.argmax(target) == 0:
    print("Ham")
elif np.argmax(target) == 1:
    print("Spam")

with open('tokenizer_two.pickle', 'wb') as tk:
    pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)
model_json = model.to_json()
with open("model_two.json", "w") as js:
    js.write(model_json)

model.save_weights("model_two.h5")
