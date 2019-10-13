from keras.models import load_model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import stanfordnlp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# def clean_text(text):
    
#     ## Remove puncuation
#     text = text.translate(string.punctuation)
    
#     ## Convert words to lower case and split them
#     text = text.lower().split()
    
#     ## Remove stop words
#     # stops = set(stopwords.words("english"))
#     # text = [w for w in text if not w in stops and len(w) >= 3]
    
#     text = " ".join(text)

#     # Clean the text
#     text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
#     text = re.sub(r"what's", "what is ", text)
#     # text = re.sub(r"\'s", " ", text)
#     # text = re.sub(r"\'ve", " have ", text)
#     text = re.sub(r"n't", " not ", text)
#     text = re.sub(r"i'm", "i am ", text)
#     text = re.sub(r"\'re", " are ", text)
#     text = re.sub(r"\'d", " would ", text)
#     text = re.sub(r"\'ll", " will ", text)
#     text = re.sub(r",", " ", text)
#     # text = re.sub(r"\.", " ", text)
#     text = re.sub(r"!", " ! ", text)
#     text = re.sub(r"\/", " ", text)
#     text = re.sub(r"\^", " ^ ", text)
#     text = re.sub(r"\+", " + ", text)
#     text = re.sub(r"\-", " - ", text)
#     text = re.sub(r"\=", " = ", text)
#     text = re.sub(r"'", " ", text)
#     text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
#     text = re.sub(r":", " : ", text)
#     text = re.sub(r" e g ", " eg ", text)
#     text = re.sub(r" b g ", " bg ", text)
#     text = re.sub(r" u s ", " american ", text)
#     text = re.sub(r"\0s", "0", text)
#     # text = re.sub(r" 9 11 ", "911", text)
#     # text = re.sub(r"e - mail", "email", text)
#     # text = re.sub(r"j k", "jk", text)
#     text = re.sub(r"\s{2,}", " ", text)
    
#     # text = text.split()
#     # stemmer = SnowballStemmer('english')
#     # stemmed_words = [stemmer.stem(word) for word in text]
#     # text = " ".join(stemmed_words)

#     return text

# def test_on_file():
#     f11 = open('BNC_first_3000_non_ell.txt', 'r')
#     f21 = open('Cases_Ellipsis.txt', 'r')
#     Pairs = []
#     number_ellipsis = 0
#     number_no_ellipsis = 0
#     for line in f11:
#         Pairs.append(0)
#         number_no_ellipsis += 1
#     for line in f21:
#         Pairs.append(1)
#         number_ellipsis += 1
#     return Pairs

# f1 = open('BNC_first_1500_non_ell.txt', 'r')
# f2 = open('Cases_Ellipsis.txt', 'r')
# # f1 = open('Test_Cases.txt', 'r')
# text = f2.read()
# text = text.replace("\n\n", "\n")
# text = text.replace("\n\n\n", "\n")
# text = text.split("\n")
# # text = f2.read().split("\n")
# Pairs = []
# nlp = stanfordnlp.Pipeline() 
# i=0
# while(i< len(text)):
#     if(text[i]=="\n" or text[i]==""):
#         text.remove(text[i])
#     else:
#         i += 1
#         Pairs.append(1)
# print("Text length: ")
# print(len(text))
# print(len(Pairs))
# text1 = text
# text = f1.read()
# text = text.replace("\n\n", "\n")
# text = text.replace("\n\n\n", "\n")
# text = text.split("\n")
# # text = f2.read().split("\n")
# i=0
# while(i< len(text)):
#     if(text[i]=="\n" or text[i]==""):
#         text.remove(text[i])
#     else:
#         i += 1
#         Pairs.append(0)
# text = text1 + text
# print("Text length: ")
# print(len(text))
# print(len(Pairs))

# for i in range(0, len(text)):
# 	# text[i] = clean_text(text[i]).strip()
# 	line = text[i]
# 	if(line != ""):
# 		POS = ""
# 		doc = nlp(line)
# 		for sent in doc.sentences:
# 			for word in sent.words:
# 				POS = POS + word.pos + " "
# 		line = line + " "+ POS
# 		text[i] = line
# 		# print(text[i])
# #vocabulary_size = 20000
# #tokenizer = Tokenizer(num_words= vocabulary_size)
# #tokenizer.fit_on_texts(text)
# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
# sequences = tokenizer.texts_to_sequences(text)
# data = pad_sequences(sequences, maxlen=120)
# print(data.shape)

# model = load_model('my_model.h5')
# Res = model.predict_classes(data)


# for i in range(len(Res)):
#     print(text[i])
#     print("Returned: ") 
#     print(Res[i]) 
#     print("Actual: ")
#     print(Pairs[i])

# # true_positive = 0
# # false_positive = 0
# # true_negative = 0
# # false_negative = 0
# # Pairs = test_on_file()
# C = confusion_matrix(Pairs, Res, labels = None, sample_weight = None)
# for row in C:
#     print(row)
# precision = C[1][1]/(C[0][1] + C[1][1])
# recall = C[1][1]/(C[1][1] + C[1][0])
# f1_score = f1_score(Pairs, Res, labels = None, sample_weight = None)
# print(precision)
# print(recall)
# print(f1_score)






# #     if(Pairs[i] == 1):
# #         if(Res[i] == 1):
# #             true_positive += 1
# #         else:
# #             false_negative += 1
# #     else:
# #         if(Res[i] == 1):
# #             false_positive += 1
# #         else:
# #             true_negative += 1
    
# # print("true_positive: ", true_positive)
# # print("true_negative: ", true_negative)
# # print("false_positive: ", false_positive)
# # print("false_negative: ", false_negative)
# # total = true_positive + true_negative + false_positive + false_negative
# # print(total)



f1 = open('Cases_Ellipsis_BNC.txt', 'r')
f2 = open('BNC_first_3000_non_ell.txt', 'r')
f3 = open('Cases_Ellipsis_BNC_withPOS.txt', 'r')
f4 = open('NC_first_3000_non_ell_withPOS.txt', 'r')
f1_split = f1.read().split("\n")

# print(f1_split)
f2_split = f2.read().split("\n")
# print(f2_split)
f3_split = f3.read().split("\n")
f4_split = f4.read().split("\n")
print("f3 split")
print(len(f3_split))
data = []
# nlp = stanfordnlp.Pipeline() 
# for line in f1_split:
#     if(line != "" and line != "\n"):
#         POS = ""
#         doc = nlp(line)
#         for sent in doc.sentences:
#             for word in sent.words:
#                 POS = POS + word.pos + " "
#         line = line + " "+ POS
#         print(line, file = f3)
#         data.append([1, line]) 

# for line in f2_split:
#     if(line != "" and line != "\n"):
#         POS = ""
#         doc = nlp(line)
#         for sent in doc.sentences:
#             for word in sent.words:
#                 POS = POS + word.pos + " "
#         line = line + " "+ POS
#         print(line, file = f4)
#         data.append([0, line])
for line in f3_split:
    if(line != "" and line != "\n"):
        data.append(line) 
for line in f4_split:
    if(line != "" and line != "\n"):
        data.append(line) 
#Preparing test data
test_data = data[:500] + data[6000:]
print("test data: ")
print(len(test_data))

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(test_data)
test_sequences = tokenizer.texts_to_sequences(test_data)
data = pad_sequences(test_sequences, maxlen=120)
print(len(test_data))
print("test data shape")
print(data.shape)
print(data[0])

feature_file = open("feature_file.txt", "r")
feature = []
for line in feature_file:
    feature.append(line.strip().split())

#Preparing test features
test_feature = feature[:500] + feature[6000:]
Actual = [1]*500
Actual += [0]*563
tokenizer_test_features = Tokenizer(num_words=4)
tokenizer_test_features.fit_on_texts(test_feature)
sequences_test_features = tokenizer_test_features.texts_to_sequences(test_feature)
test_features = pad_sequences(sequences_test_features, maxlen=6)
print("Test features shape: ")
print(test_features.shape)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Res = []
# print(len(test_data))
model = load_model('my_model.h5')
# for i in range(len(data)):
#     x = model.predict([data, test_features])
#     print(x)
#     Res.append(np.argmax(x,axis=1))


Results = model.predict([data, test_features])
print(len(Results))
print(Results[0][0])
Res = []
for i in range(len(Results)):
    Res.append(round(Results[i][0]))

print(len(Res))
for i in range(len(Res)):
    print(test_data[i])
    print("Returned: ") 
    print(Res[i]) 
    print("Actual: ")
    print(Actual[i])

C = confusion_matrix(Actual, Res, labels = None, sample_weight = None)
for row in C:
    print(row)
precision = C[1][1]/(C[0][1] + C[1][1])
recall = C[1][1]/(C[1][1] + C[1][0])
f1_score = f1_score(Actual, Res, labels = None, sample_weight = None)
print(precision)
print(recall)
print(f1_score)



