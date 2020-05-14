# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
from collections import Counter
import string

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re
import nltk

import torch
from torchtext.data import Example, Dataset, Field, LabelField, TabularDataset, Iterator, BucketIterator

nltk.download('punkt')
nltk.download('wordnet')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocessing_factory(functions, language='english'):
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language)
    stop = stopwords.words(language)
    
    def remove_punc(text):
        return re.sub(r"[^A-Za-z0-9]", " ", str(text))
    
    def remove_punc_2(text):
        text = text.replace("(", " ").replace(")", " ")
        text = re.sub(' +', ' ', text)
        text = re.sub("(\w+)\.([A-Z]+)", "\g<1> \g<2>", text)
        return text
    
    def remove_xml(text):
        return BeautifulSoup(text, "lxml").get_text()
    
    def lower_text(text):
        return text.lower()
    
    def simple_tokenize(text):
        return text.split()
    
    def advanced_tokenize(text):
        return word_tokenize(text)
    
    def remove_stopwords(text):
        return [w for w in text if not w in stop]
    
    def lemmatizing(text):
        return [lemmatizer.lemmatize(word) for word in text]
    
    def stemming(text):
        return [stemmer.stem(word) for word in text]
    
    def nothing(text):
        return text
    
    convert = {'punct':remove_punc, 'custPunct':remove_punc_2,
               'xml':remove_xml, 'lower':lower_text,
               'simpleToken':simple_tokenize, 'advToken':advanced_tokenize,
               'stopwords':remove_stopwords,
               'lemming':lemmatizing, 'stemming':stemming,
               'nothing':nothing}
    
    return [convert[func] for func in functions]

def apply_all(text, functions):
    for func in functions:
      text = func(text)
    return text

def preprocess(reviews, functions):
    tokens = list(map(lambda x: apply_all(x, preprocessing_factory(filter(lambda x: x!='nothing', functions))), reviews))
    return tokens

def common_percentage(tokens, vocab):
  return sum([w in vocab for sentence in tokens for w in sentence])/sum(map(len, tokens))

def check_difference(tokens, vocab):
  return list(set([w for sentence in tokens for w in sentence if w not in vocab]))




def clean_quora(path='../data/train.csv', output='list', tokenizer = nltk.word_tokenize, device=DEVICE, batch_size=32):
    data = pd.read_csv(path)
    questions1 = data['question1'].astype('str').tolist()
    questions2 = data['question2'].astype('str').tolist()
    is_duplicates = data['is_duplicate'].tolist()
    
    if output == 'list':
        return questions1, questions2, is_duplicates
    
    elif output == 'tokenized_list':
        return [tokenizer(q) for q in questions1], [tokenizer(q) for q in questions2], is_duplicates
    
    elif output == 'iterator' or output == 'iterator_from_file':
        TEXT = Field(
                sequential=True,
                tokenize = tokenizer,
                pad_first = False,
                dtype = torch.long,
                lower = True,
                batch_first = True
                )
        TARGET = LabelField(use_vocab = False)
        
        if output == 'iterator':
            examples = [Example.fromlist((questions1[i], questions2[i], is_duplicates[i]),
                                         [('question1', TEXT),
                                          ('question2', TEXT)
                                          ('is_duplicate', TARGET)]) for i in range(len(questions1))]
            dataset = Dataset(examples, {'question1': TEXT, 'question2': TEXT, 'is_duplicate': TARGET})
    
        if output == 'iterator_from_file':
            dataset = TabularDataset(path, 'csv', [('question1', TEXT),
                                                   ('question2', TEXT),
                                                   ('is_duplicate', TARGET)],
    skip_header=True)
        
        iterator = BucketIterator(
                dataset,
                batch_size=batch_size,
                sort_key=lambda x: len(x.question1) + len(x.question2),
                sort_within_batch=False,
                repeat = False,
                device = device
                # repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
        )
        
        TEXT.build_vocab(dataset)
        TARGET.build_vocab(dataset)
        
        
        return iterator
        

        #dataset = TabularDataset(path, 'csv', [('review', TEXT), ('sentiment', TARGET)])
        
    else:
        raise ValueError('Processing type not understood')

    
