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

feature_file = open("feature_file.txt", "r")
feature = []
for line in feature_file:
	integer_features = [int(x) for x in line.strip().split()]
	feature.append(integer_features)
test_feature = feature[:500] + feature[6063:]
train_feature = feature[500:6063]
#Labels
train_labels = []
for i in range(3062):
	train_labels.append(1)
for i in range(2501):
	train_labels.append(0)
#test features
test_labels = [1]*500
test_labels += [0]*500
model_feature_1 = Input(shape = (6,))
output = Dense(1, activation='sigmoid')(model_feature_1)
model_feature = Model(inputs = model_feature_1, outputs = output)
model_feature.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_feature.fit(np.array(train_feature), np.array(train_labels), validation_split=0.2, epochs = 10)
model_feature.save('my_model_only_features.h5')  # creates a HDF5 file 'my_model.h5'
model = load_model('my_model_only_features.h5')
Results = model.predict(np.array(test_feature))
# print(len(Results))
# print(Results[0][0])
Res = []
for i in range(len(Results)):
    Res.append(round(Results[i][0]))

print(len(Res))
# for i in range(len(Res)):
#     print(test_data[i])
#     print("Returned: ") 
#     print(Res[i]) 
#     print("Actual: ")
#     print(test_labels[i])

C = confusion_matrix(test_labels, Res, labels = None, sample_weight = None)
for row in C:
    print(row)
precision = C[1][1]/(C[0][1] + C[1][1])
recall = C[1][1]/(C[1][1] + C[1][0])
f1_score = f1_score(test_labels, Res, labels = None, sample_weight = None)
print(precision)
print(recall)
print(f1_score)
