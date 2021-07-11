#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk   #natural language toolkit
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

import tkinter
from tkinter import *
# In[2]:


import tensorflow as tf
import numpy as np
import tflearn
import random
import json
from tensorflow.python.framework import ops


# In[3]:


with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[4]:


intents


# In[5]:


words=[]
classes=[]
documents=[]
ignore=['?']
#loop through each sentence in the intent's patters
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each and every word in the sentence
        w=nltk.word_tokenize(pattern)
        #add word to the words list
        words.extend(w)
        #add word(s) to documents
        documents.append((w,intent['tag']))
        #add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[6]:


#perform stemming and lower each word as well as remove duplicates
words=[stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

#remove duplicate classes
classes= sorted(list(set(classes)))

print(len(documents),"documents")
print(len(classes),"classes", classes)
print(len(words),"unique stemmed words", words)
documents


# In[7]:


#create training data
training=[]
output=[]
#create an empty array for output
output_empty = [0]*len(classes)

#create training set, bag of words for each sentence
for doc in documents:
    #initialize bag of words
    bag=[]
    #list of tokenized words for the pattern
    pattern_words=doc[0]
    #stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    #create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    #output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    
    output_row[classes.index(doc[1])]=1
   # print(output_row)
    training.append([bag,output_row])

#print(training)
#shuffeling features and turning it into np.array
random.shuffle(training)
training=np.array(training)
#creating training lists
train_x=list(training[:,0])
train_y=list(training[:,1])


# In[8]:


#resetting underlying graph data

ops.reset_default_graph()

#building neural network
net= tflearn.input_data(shape=[None, len(train_x[0])])
net= tflearn.fully_connected(net,10)
net= tflearn.fully_connected(net,10)
net= tflearn.fully_connected(net,len(train_y[0]),activation='softmax')
net= tflearn.regression(net)

#Defining model and setting up tensorboard
model =  tflearn.DNN(net, tensorboard_dir='tflearn_logs')

#Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# In[9]:


import pickle
pickle.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open("training_data","wb"))


# In[10]:


# restoring all the data structures

data = pickle.load( open("training_data","rb"))
words=data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


# In[11]:


with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[12]:


# load the saved model
model.load('./model.tflearn')


# In[13]:


def clean_up_sentence(sentence):
    #tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    #stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# returning bag of words array :0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=False):
    #tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    #generating bag od words
    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("found in bag: %s" %w)
    return(np.array(bag))


# In[14]:


ERROR_THRESHOLD = 0.30
def classify(sentence):
    #generate probabilities from the model
    results = model.predict([bow(sentence,words)])[0]
    # filter our predictions below a threshold
    results = [[i,r] for i, r in enumerate(results) if r>ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append((classes[r[0]],r[1]))
        #return tuple of intent and probability
        return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return (random.choice(i['responses']))

            results.pop(0)







def chatbot_response(msg):

    res = response(msg)

    return res+'\n'

def send():
    msg = TextEntryBox.get("1.0", 'end-1c').strip()
    TextEntryBox.delete('1.0', 'end')

    if msg != '':
        ChatHistory.config(state=NORMAL)
        ChatHistory.insert('end', "You: " + msg + "\n\n")

        res = chatbot_response(msg)
        ChatHistory.insert('end', "Bot: " + res+"\n\n")
        ChatHistory.config(state=DISABLED)
        ChatHistory.yview('end')

base = Tk()
base.title("Alexa")
base.geometry("400x500")
base.resizable(width=False, height=False)

#chat history textview
ChatHistory = Text(base, bd=0, bg='white', font='Arial')
ChatHistory.config(state=DISABLED)

SendButton = Button(base, font=('Arial', 12, 'bold'), 
    text="Send", bg="#dfdfdf", activebackground="#3e3e3e", fg="#ffffff", command=send)

TextEntryBox = Text(base, bd=0, bg='white', font='Arial')

ChatHistory.place(x=6, y=6, height=386, width=386)
TextEntryBox.place(x=128, y=400, height=80, width=265)
SendButton.place(x=6, y=400, height=80, width=125)

base.mainloop()