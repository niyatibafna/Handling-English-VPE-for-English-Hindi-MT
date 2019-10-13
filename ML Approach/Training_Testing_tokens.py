from keras.models import load_model
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle
import stanfordnlp
from keras.layers import *

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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
f1 = open('Cases_Ellipsis_BNC.txt', 'r')
f2 = open('BNC_first_3000_non_ell.txt', 'r')
f1_split = f1.read().split("\n")
# print(f1_split)
f2_split = f2.read().split("\n")
# print(f2_split)
data = []
for line in f1_split:
    if(line != "" and line != "\n"):
        data.append([1, line]) 

for line in f2_split:
    if(line != "" and line != "\n"):
        data.append([0, line])
#Preparing test data
test_data_labelled = data[:500] + data[6000:]
test_data = []
for i in range(len(test_data_labelled)):
    test_data.append(test_data_labelled[i][1])

data = data[500:6000]
# print(len(test_data))
df = pd.DataFrame(data, columns = ['ellipsis', 'text'])
print(df.head())

#Test data
test_df = pd.DataFrame(test_data_labelled, columns = ['test_ellipsis', 'test_text'])
print(test_df.head())

labels = df['ellipsis'].map(lambda x : 1 if int(x) == 1  else 0)
df['text'] = df['text'].map(lambda x: x)
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=120)
print("Train data shape: ")
print(data.shape)

#Test data
test_labels = test_df['test_ellipsis'].map(lambda x : 1 if int(x) == 1  else 0)
test_df['test_text'] = test_df['test_text'].map(lambda x: x)
vocabulary_size = 20000
test_tokenizer = Tokenizer(num_words= vocabulary_size)
test_tokenizer.fit_on_texts(test_df['test_text'])
test_sequences = test_tokenizer.texts_to_sequences(test_df['test_text'])
test_data = pad_sequences(test_sequences, maxlen=120)
print("Test data shape: ")
print(test_data.shape)



# embeddings_index = dict()
# f = open('glove.6B/glove.6B.100d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))

# # create a weight matrix for words in training docs
# embedding_matrix = np.zeros((vocabulary_size, 100))
# for word, index in tokenizer.word_index.items():
#     if index > vocabulary_size - 1:
#         break
#     else:
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[index] = embedding_vector


# del embeddings_index
# model_glove = Sequential()
model_feature_1 = Input(shape = (120,))
output = Dense(1, activation='sigmoid')(model_feature_1)
model_feature = Model(inputs = model_feature_1, outputs = output)
model_feature.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_feature.fit(data, np.array(df['ellipsis']), validation_split=0.2, epochs = 10)
model_feature.save('my_model_only_tokens.h5')  # creates a HDF5 file 'my_model.h5'
model = load_model('my_model_only_tokens.h5')
Results = model.predict(np.array(test_data))
Res = []
for i in range(len(Results)):
    Res.append(round(Results[i][0]))

C = confusion_matrix(np.array(test_df['test_ellipsis']), Res, labels = None, sample_weight = None)
for row in C:
    print(row)
precision = C[1][1]/(C[0][1] + C[1][1])
recall = C[1][1]/(C[1][1] + C[1][0])
f1_score = f1_score(np.array(test_df['test_ellipsis']), Res, labels = None, sample_weight = None)
print(precision)
print(recall)
print(f1_score)


# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# model = load_model('my_model.h5')
# Res = model.predict_classes([test_data, test_features])


# for i in range(len(Res)):
#     print(test_data[i])
#     print("Returned: ") 
#     print(Res[i]) 
#     print("Actual: ")
#     print(Actual[i])

# true_positive = 0
# false_positive = 0
# true_negative = 0
# false_negative = 0
# Pairs = test_on_file()
# C = confusion_matrix(Actual, Res, labels = None, sample_weight = None)
# for row in C:
#     print(row)
# precision = C[1][1]/(C[0][1] + C[1][1])
# recall = C[1][1]/(C[1][1] + C[1][0])
# f1_score = f1_score(Actual, Res, labels = None, sample_weight = None)
# print(precision)
# print(recall)
# print(f1_score)

