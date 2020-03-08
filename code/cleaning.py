#!/usr/bin/env python
# coding: utf-8

# In[82]:


# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:09:04 2020

@author: clara
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
import re
from string import punctuation
import matplotlib.pyplot as plt


# In[83]:


df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")


# In[84]:


print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()


# In[88]:


#stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
#              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
#              'Is','If','While','This']

def text_to_wordlist(text, remove_stop_words=True, lemmatize=True,stem_words=True):
    # Clean the text, with the option to remove stop_words lemmatize and to stem words.
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text))

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    stopWords = set(stopwords.words('english'))
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stopWords]
        text = " ".join(text)
        
        
    #Optionally, Lemmatize words
    lemmatizer = WordNetLemmatizer() 
    if lemmatize:
        text = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
 g

# In[89]:


def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


# In[90]:


train_question1 = []
process_questions(train_question1, df_train.question1, 'train_question1', df_train)

train_question2 = []
process_questions(train_question2, df_train.question2, 'train_question2', df_train)


# In[92]: 


a = 0 
for i in range(a,a+2):
    print(df_train.question1[i])
    print(df_train.question2[i])
    print()

a = 0 
for i in range(a,a+2):
    print(train_question1[i])
    print(train_question2[i])
    print()

