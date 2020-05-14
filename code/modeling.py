#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:30:42 2020

@author: henric
"""
import numpy as np

import xgboost as xgb

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras import layers

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

def get_simple_nn_model(input_dim, hidden_sizes=[32, 16, 8]):
  model = keras.Sequential()
  for i, hidden in enumerate(hidden_sizes):
    if i==0:
      model.add(layers.Dense(hidden, activation='relu', input_shape=(input_dim, )))
    else:
      model.add(layers.Dense(hidden, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])

  return model

def model_efficiency(embs, labels, model=LogisticRegression(), validation=False, reinitialize=True, **params):
  X_train, X_valid, y_train, y_valid = train_test_split(embs, labels, train_size = 0.7, random_state=42, shuffle=True, stratify=labels)
  
  y_train, y_valid = np.array(y_train), np.array(y_valid)
  
  if issubclass(type(model), (tensorflow.python.keras.engine.sequential.Sequential, tensorflow.keras.Model)):
      params['validation_data'] = (X_valid, y_valid)

  if reinitialize:
    if issubclass(type(model), (tensorflow.python.keras.engine.sequential.Sequential, tensorflow.keras.Model)):
      model_copy = keras.models.clone_model(model)
      model_copy.build((None, model.input.shape))
      model_copy.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
      model = model_copy
    else:
        try:
            model = clone(model)
        except Exception:
            print('model not reinitialized')
            pass
      
  model.fit(X_train, y_train, **params)
  
  if hasattr(model, 'predict_proba'):
      proba_predictions_train = model.predict_proba(X_train)
      proba_predictions_valid = model.predict_proba(X_valid)
  else:
      proba_predictions_train = model.predict(X_train)
      proba_predictions_valid = model.predict(X_valid)
          
  if proba_predictions_train.shape[1]==2:
      predictions_train = proba_predictions_train.argmax(axis=1)
      predictions_valid = proba_predictions_valid.argmax(axis=1)
  else:
      predictions_train = proba_predictions_train >= 0.5
      predictions_valid = proba_predictions_valid >= 0.5

  loss_train = log_loss(y_train, proba_predictions_train)
  loss_valid = log_loss(y_valid, proba_predictions_valid)

  accuracy_train = accuracy_score(y_train, predictions_train)
  accuracy_valid = accuracy_score(y_valid, predictions_valid)

  f1_train = f1_score(y_train, predictions_train)
  f1_valid = f1_score(y_valid, predictions_valid)

  return {'loss':np.round(loss_train,2), 'accuracy':np.round(accuracy_train,2), 'f1':np.round(f1_train,2)}, \
         {'loss':np.round(loss_valid,2), 'accuracy':np.round(accuracy_valid,2), 'f1':np.round(f1_valid,2)}
         
         
def model_efficiency_xgb(embs, labels, params={'objective':'binary:logistic', 'eval_metric':'logloss', 'eta':0.02, 'max_depth':4}):

  X_train, X_valid, y_train, y_valid = train_test_split(embs, labels, train_size = 0.7, random_state=42, shuffle=True, stratify=labels)

  d_train = xgb.DMatrix(X_train, label=y_train)
  d_valid = xgb.DMatrix(X_valid, label=y_valid)
  watchlist = [(d_train, 'train'), (d_valid, 'valid')]

  model = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

  proba_predictions_train = model.predict(d_train)
  proba_predictions_valid = model.predict(d_valid)

  predictions_train = proba_predictions_train >= 0.5
  predictions_valid = proba_predictions_valid >= 0.5

  loss_train = log_loss(y_train, proba_predictions_train)
  loss_valid = log_loss(y_valid, proba_predictions_valid)

  accuracy_train = accuracy_score(y_train, predictions_train)
  accuracy_valid = accuracy_score(y_valid, predictions_valid)

  f1_train = f1_score(y_train, predictions_train)
  f1_valid = f1_score(y_valid, predictions_valid)

  return {'loss':np.round(loss_train,2), 'accuracy':np.round(accuracy_train,2), 'f1':np.round(f1_train,2)}, \
         {'loss':np.round(loss_valid,2), 'accuracy':np.round(accuracy_valid,2), 'f1':np.round(f1_valid,2)}