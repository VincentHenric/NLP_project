#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:03:25 2020

@author: henric
"""
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import subprocess
import sys
import re
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def download_and_install_git(url):
    search = re.search('([a-zA-Z0-9_]*)/archive/', url)
    if search:
        folder_name = search.group(1)
    else:
        raise ValueError('no match')
    if not os.path.isdir('{}-master'.format(folder_name)):
        subprocess.check_call(['wget', url])
        subprocess.check_call(['unzip', 'master.zip'])
    install('./{}-master/.'.format(folder_name))
    print('package installed')
    
class embedding_model:
    def __init__(kind, path):
        pass
    
    def load_model():
        pass
    
    def calculate_embeddings():
        pass
    

class Doc2Vec_model(embedding_model):
    def __init__(self, path='../embeddings/doc2vec/enwiki_dbow/doc2vec.bin'):
        self.path = path
        
    def load_model(self):
        from gensim.models.doc2vec import Doc2Vec
        self.model = Doc2Vec.load(self.path)
        
    def calculate_embeddings(self, tokenized_list):
        embs = np.zeros((len(tokenized_list), 300))
        for i, text in enumerate(tokenized_list):
          embs[i,:] = self.model.infer_vector(text)
        return embs
        
class Sent2Vec(embedding_model):
    def __init__(self, path='../embeddings/sent2vec/sent2vec_toronto_books_unigrams'):
        self.path = path
        download_and_install_git('https://github.com/epfml/sent2vec/archive/master.zip')
        
    def load_model(self):
        import sent2vec
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(self.path)
        
    def calculate_embeddings(self, list):
        embs = self.model.embed_sentences(list)
        return embs
    
class SBERT(embedding_model):
    def __init__(self, path='bert-base-nli-mean-tokens'):
        install('sentence-transformers')
        self.path = path
        
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.path)
        
    def calculate_embeddings(self, list):
        embs = self.model.encode(list)
        return embs
    
class SIF(embedding_model):
    def __init__(self, path="glove-wiki-gigaword-100"):
        install('fse')
        self.path = path
        
    def load_model(self, language='en'):
        import gensim.downloader as api
        from fse.models import uSIF
        from fse import IndexedList
        word_embedding = api.load(self.path)
        self.model = uSIF(word_embedding, lang_freq=language)
        
    def fit(self, list):
        texts = fse.IndexedList(list)
        self.model.train(texts)
        
    def calculate_embeddings(self, list):
        texts = fse.IndexedList(list)
        embs = self.model.infer(texts)
        return embs

class Count(embedding_model):
    def __init__(self, **kwargs):
        self.params = {'encoding': 'unicode',
                       'stop_words': stopwords.words('english'),
                       'tokenizer': lambda x: x.split(' '),
                       'min_df': 2}
        self.params.update(kwargs)
        
    def load_model(self, **kwargs):
        self.params.update(kwargs)
        self.model = CountVectorizer(**self.params)
        
    def fit(self, list):
        self.model.fit(list)
        
    def calculate_embeddings(self, list):
        return self.model.transform(list)
        
        
class TFIDF(embedding_model):
    def __init__(self, **kwargs):
        self.params = {'encoding': 'unicode',
                       'stop_words': stopwords.words('english'),
                       'tokenizer': lambda x: x.split(' '),
                       'min_df': 2}
        self.params.update(kwargs)
        
    def load_model(self, **kwargs):
        self.params.update(kwargs)
        self.model = TfidfVectorizer(**self.params)
        
    def fit(self, list):
        self.model.fit(list)
        
    def calculate_embeddings(self, list):
        return self.model.transform(list)