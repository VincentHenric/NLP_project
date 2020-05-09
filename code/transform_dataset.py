# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import pandas as pd

import nltk

import torch
from torchtext.data import Example, Dataset, Field, LabelField, TabularDataset, Iterator, BucketIterator

nltk.download('punkt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    