import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
stemm = LancasterStemmer()

words = []
docs_x = []
docs_y = []
classes = []
with open("girlfriend.json") as f:
    data = json.load(f)
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
words = [stemm.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))
classes = sorted(classes)
training = []
output = []
out_empty = [0 for _ in range(len(classes))]#[0,0,0]
for x, doc in enumerate(docs_x):#[[],[]]
    bag = []
    wrds = [stemm.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[classes.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

output = numpy.array(output)
training = numpy.array(training)
with open("data.pickle", "wb") as f:
    pickle.dump((words, classes, training, output), f)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("girlfriend.tflearn")









