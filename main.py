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

# load dataset from csv file and use encoding as latin-1
email_df = pd.read_csv("spam.csv", encoding='latin-1')
# print first 5 records
print(email_df.head())
# print concise summary about dataset.
print(email_df.info())
# lets delete Unnamed : 2 , Unnamed : 3 and Unnamed : 4 column because they are having zero values in almost entire column.
column_to_delete = [name for name in email_df.columns if name.startswith('Unnamed')]
email_df.drop(columns=column_to_delete, inplace=True)

# rename v1 column to target and v2 column to message
email_df.rename(columns=dict({"v1": "target", "v2": "message"}), inplace=True)

print(email_df.tail())

# print null values
email_df.isnull().sum()

# print no of duplicate records
print("Total duplicated records in dataset are : {}".format(email_df.duplicated().sum()))

# lets remove duplicated records
email_df.drop_duplicates(inplace=True)

email_df['message'] = email_df['message'].apply(lambda x: x.lower())
email_df['message'] = email_df['message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(email_df.tail())

max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(email_df['message'].values)
X = tokenizer.texts_to_sequences(email_df['message'].values)
X = pad_sequences(X)
print(X.shape)

embed_dim = 50
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(LSTM(10))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

y = pd.get_dummies(email_df['target']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model.fit(X_train, y_train, epochs=5, verbose=1)

test = ['Nah I dont think he goes to usf, he lives around here though']
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, maxlen=X.shape[1], dtype='int32', value=0)
print(test.shape)
target = model.predict(test)[0]
if np.argmax(target) == 0:
    print("Ham")
elif np.argmax(target) == 1:
    print("Spam")

with open('tokenizer.pickle', 'wb') as tk:
    pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)
model_json = model.to_json()
with open("model.json", "w") as js:
    js.write(model_json)

model.save_weights("model.h5")
