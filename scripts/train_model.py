# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 15:08:23 2017

@author: mariapanteli
"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd

import models
import util_model

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# download from http://c4dm.eecs.qmul.ac.uk/worldmusicoutliers/BLSF_corpus/
METADATA_FILE = '../data/metadata_BLSF_corpus.csv'  
HDF5_PATH = '../data/melspec_BLSF_corpus.hdf5'
SEED = 12345  # for reproducibility
MODEL_LABEL = 'cnn_4L'  # cnn_4L is best, otherwise choose 'cnn_2L' or 'mfcc_nn'
BASENAME = '../data/melspec_world_tagging_' + str(MODEL_LABEL) 
BATCH_SIZE = 16  # increase depending on gpu
EPOCHS = 100

X, Y, tags = util_model.subset_tags(METADATA_FILE, HDF5_PATH, seed=SEED)
print(X.shape, Y.shape)
if MODEL_LABEL == 'mfcc_nn':
    # extract mfccs from melspec
    X = util_model.mfcc_data(X)

(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = util_model.get_train_val_test_idx(X, Y, seed=SEED)
# check tags are well distributed in train, val, and test sets
print(list(zip(tags, np.sum(Y_train, axis=0))))
print(list(zip(tags, np.sum(Y_val, axis=0))))
print(list(zip(tags, np.sum(Y_test, axis=0))))

if MODEL_LABEL=='cnn_2L':
    model = models.cnn_2L((X.shape[1], X.shape[2], X.shape[3]), len(tags))
elif MODEL_LABEL=='cnn_4L':
    model = models.cnn_4L((X.shape[1], X.shape[2], X.shape[3]), len(tags))
elif MODEL_LABEL=='mfcc_nn':
    model = models..mfcc_nn((X.shape[1:]), len(tags))
else:
    raise ValueError('Model label must exist in models.py choose cnn_2L, cnn_4L, or mfccnn')
    
model = util_model.train_model(model, X_train, Y_train, X_val, Y_val, tags, 
                                     batch_size=BATCH_SIZE, epochs=EPOCHS)
util_model.write_model(model, BASENAME)
auc, pred_tags = util_model.test_model(model, X_test, Y_test)

