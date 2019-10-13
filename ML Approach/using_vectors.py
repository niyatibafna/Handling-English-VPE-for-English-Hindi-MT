from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle
import stanfordnlp

## Plot
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import matplotlib as plt

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    # stops = set(stopwords.words("english"))
    # text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)

    return text
f1 = open('Cases_Ellipsis_WSJ.txt', 'r')
f2 = open('Cases_No_Ellipsis.txt', 'r')
f1_split = f1.read().split("\n")
# print(f1_split)
f2_split = f2.read().split("\n")
# print(f2_split)
data = []
nlp = stanfordnlp.Pipeline() 
for line in f1_split:
    if(line != ""):
        POS = ""
        doc = nlp(line)
        for sent in doc.sentences:
            for word in sent.words:
                POS = POS + word.pos + " "
        line = line + " "+ POS
        data.append([1, line]) 

for line in f2_split:
    if(line != ""):
        POS = ""
        doc = nlp(line)
        for sent in doc.sentences:
            for word in sent.words:
                POS = POS + word.pos + " "
        line = line + " "+ POS
        data.append([0, line])

df = pd.DataFrame(data, columns = ['ellipsis', 'text'])
print(df.head())

labels = df['ellipsis'].map(lambda x : 1 if int(x) == 1  else 0)
df['text'] = df['text'].map(lambda x: x)
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])


#Creating features


sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=120)
print(data.shape)

# model_lstm = Sequential()
# model_lstm.add(Embedding(20000, 100, input_length=50))
# model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model_lstm.add(Dense(1, activation='softmax'))
# model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model_lstm.fit(data, np.array(labels), validation_split=0.2, epochs=3)

embeddings_index = dict()
f = open('glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 100, input_length=120, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.1))
model_glove.add(Conv1D(128, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(200))

model_glove.add(Embedding(vocabulary_size, 100, input_length=120, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.1))
model_glove.add(Conv1D(128, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(200))

model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.fit(data, np.array(labels), validation_split=0.2, epochs = 10)
model_glove.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

