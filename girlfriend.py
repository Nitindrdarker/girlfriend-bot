import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()#root word of a word

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("girlfriend.json") as file:
    data = json.load(file)
with open("data.pickle","rb") as f:
    words, classes, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("girlfriend.tflearn")
def bag_of_words(s_input):
    bag_of_input = [0 for _ in range(len(words))]
    input_words = nltk.word_tokenize(s_input)
    input_words = [stemmer.stem(w.lower()) for w in input_words]
    for se in input_words:
        for i, w in enumerate(words):
            if w == se:
                bag_of_input[i] = 1
    return numpy.array(bag_of_input)
            
def chat():
    print("bot running!")
    while True:
        inp = input("you:")
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp)])[0]
        result_index = numpy.argmax(result)
        clas = classes[result_index]
        if result[result_index] > 0.5:
            for tag in data["intents"]:
                if tag["tag"] == clas:
                    response = tag["responses"]
            print(random.choice(response))
        else:
            print("I Can't understand you babe!")
chat()
