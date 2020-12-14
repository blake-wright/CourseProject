#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer
import pandas as pd
from transformers import DistilBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[2]:



#read in train & test data
trainData = pd.read_json('train.jsonl', lines = True)
testData = pd.read_json('test.jsonl', lines = True)


# In[3]:


for i in range(len(trainData)):
    if trainData['label'][i] == "SARCASM":
        trainData['label'][i] = 1
    else:
        trainData['label'][i] = 0

#trainData = trainData[:5000]
#testData = testData[:1800]

#setting modelId, tokenizer, and model
modelId = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelId)
model = DistilBertModel.from_pretrained(modelId)


# In[4]:


tokenized = trainData['response'].apply((lambda x: tokenizer.encode(x, add_special_tokens = True)))
tokenized_2 = testData['response'].apply((lambda x: tokenizer.encode(x, add_special_tokens = True)))

trainPad = pad_sequences(tokenized, maxlen = 100, padding='post')
testPad = pad_sequences(tokenized_2, maxlen = 100, padding='post')

trainMask = np.where(trainPad != 0,1,0)
testMask = np.where(testPad != 0,1,0)

#converting to int64
trainInput = torch.tensor(trainPad).to(torch.int64)
testInput = torch.tensor(testPad).to(torch.int64)

#converting to tensor type
trainMask = torch.tensor(trainMask)
testMask = torch.tensor(testMask)

with torch.no_grad():
    output = model(trainInput, attention_mask = trainMask)
with torch.no_grad():
    outputTest = model(testInput, attention_mask = testMask)


# In[5]:


trainFeats = output[0][:,0,:].numpy()
testFeats = outputTest[0][:,0,:].numpy()
print(testFeats.shape)


# In[6]:


labels = trainData['label']
print(labels.shape)


# In[7]:


#splitting training and testing data
trainFeats, valFeats, trainLabels, testLabels = train_test_split(trainFeats, labels)
print(trainFeats.shape)
print(valFeats.shape)
#object types are not allowed, converting to int types
trainLabels = trainLabels.astype(str).astype(int)
testLabels = testLabels.astype(str).astype(int)


# In[15]:


classifier = RandomForestClassifier(n_estimators=500, max_depth=None,
                                         min_samples_split=8, random_state=2)
classifier.fit(trainFeats, trainLabels)
classifier.score(valFeats, testLabels)


# In[16]:


results = classifier.predict(testFeats)


# In[12]:


f = open("answer.txt", "w")

for i in range(len(results)):
    id = str(testData['id'][i])
    answer = str(results[i])
    f.write(id)
    f.write(',')
    if answer == "1":
        f.write("SARCASM")
    else:
        f.write("NOT_SARCASM")
    f.write('\n')


# In[ ]:




