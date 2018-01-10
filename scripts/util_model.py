# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 15:08:23 2017

@author: mariapanteli
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import os

from keras import backend as K
from keras.models import load_model, model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import models
import prepare_data


def load_meta_and_align(metadata_file, Y):
    """Metadata for each mel spectrogram.
    """
    df_to_align = pd.read_csv(metadata_file)
    # currently metadata['Audio'] and Y differ in the file extention
    df_to_align['Audio'] = np.array([y.split('.')[0]+'.csv' for y in df_to_align['Audio']])
    Y_df = pd.DataFrame(Y, columns=['Audio'])
    df = pd.merge(Y_df, df_to_align, on='Audio', how='left')
    df['Decade'] = np.array(np.floor(df['Year'] / 10) * 10)  # keep float otherwise nan values get assigned a random integer
    return df


def check_order(col_a, col_b):
    if len(col_a) != len(col_b):
        return False
    n_items = len(col_a)
    for i in range(n_items):
        if col_a[i] != col_b[i]:
            return False
    return True


def reshape_X(X):
    """Mel spectrograms are stored in theano dimensions.
        if backend=tensorflow reshape X.
    """
    if K.image_dim_ordering() == 'tf':
        if X.shape[3] != 1:
            X = np.rollaxis(X, 1, 4)
    return X


def mfcc_data(X):
    """Extract MFCCs for each mel-spectrogram in X

    Parameters
    ----------
    X : np.ndarray
        The tensorflow input data of dim n_tracks, n_freqs, n_frames, 1
    
    Returns
    -------
    X_mfcc : np.ndarray
        The mfcc input data in n_tracks x n_mfcc_stats
    """
    n_tracks = X.shape[0]
    n_mfcc = 20
    n_mfcc_feat = n_mfcc * 6
    X_mfcc = np.zeros((n_tracks, n_mfcc_feat))
    for i in range(n_tracks):
        mfcc_feat = prepare_data.mfcc_from_melspec(X[i, :, :, 0], n_mfcc=n_mfcc)
        X_mfcc[i, :] = mfcc_feat
    return X_mfcc


def subset_tags(metadata_file, hdf5_path, n_samples=None, min_n_tags=50, seed=None, return_df=False):
    """Load mel spectrograms and metadata, and subset tracks 
        such that each tag occurs a minimum of 50 times.
    """
    # load data
    X, Y = prepare_data.read_dataset(hdf5_path, n_samples=n_samples)
    df = load_meta_and_align(metadata_file, Y)
    print('order ', check_order(df['Audio'].iloc[:], Y))
    X = reshape_X(X)  # X in theano dim, reshape to tensorflow dim if needed
    
    # subset data
    # upper bound to 'Country' to not end up with only music from US or UK
    subset_idx = subset_labels(np.array(df['Country'], dtype=str), N_min=1, N_max=1000, seed=seed)  
    df = df.iloc[subset_idx, :].reset_index(drop=True)
    X = X[subset_idx, :]
    
    labels = np.array(df[['Country', 'Language_iso3', 'Decade']], dtype=str)
    Y_binary, tags = prepare_data.encode_labels(labels, min_n_tags=min_n_tags)
    if return_df:
        X, Y, df = prepare_data.remove_samples_with_no_tags(X, Y_binary, df=df)
        return X, Y, tags, df
    X, Y = prepare_data.remove_samples_with_no_tags(X, Y_binary)
    return X, Y, tags


def get_train_val_test_idx(X, Y, seed=None, df=None):
    """ Split in train, validation, test sets.
    
    Parameters
    ----------
    X : np.array
        Data or indices.
    Y : np.array
        Class labels for data in X.
    seed: int
        Random seed.
    Returns
    -------
    (X_train, Y_train) : tuple
        Data X and labels y for the train set
    (X_val, Y_val) : tuple
        Data X and labels y for the validation set
    (X_test, Y_test) : tuple
        Data X and labels y for the test set
    
    """
    idx_train, idx_val_test, Y_train, Y_val_test = train_test_split(np.arange(len(X)), Y, train_size=0.6, random_state=seed, stratify=Y)
    # train_test_split not suitable for multilabel data, train/test indices overlap 
    # remove overlapped indices manually 
    print(len(idx_train), len(idx_val_test))
    print('overlapping train', len([ii for ii in idx_val_test if ii in idx_train]))
    idx_val_test = np.array(list(set(idx_val_test) - set(idx_train)))
    print(len(idx_train), len(idx_val_test))
    X_train = X[idx_train, :]
    X_val_test = X[idx_val_test, :]
    Y_train = Y[idx_train, :]
    Y_val_test = Y[idx_val_test, :]
    idx_val, idx_test, Y_val, Y_test = train_test_split(np.arange(len(X_val_test)), Y_val_test, train_size=0.6, random_state=seed, stratify=Y_val_test)
    print(len(idx_val), len(idx_test))
    print('overlapping test', len([ii for ii in idx_test if ii in idx_val]))
    idx_test = np.array(list(set(idx_test) - (set(idx_val) & set(idx_test))))
    print(len(idx_val), len(idx_test))
    X_val = X_val_test[idx_val, :]
    X_test = X_val_test[idx_test, :]
    Y_val = Y_val_test[idx_val, :]
    Y_test = Y_val_test[idx_test, :]
    if df is not None:
        # return metadata for train, val, test sets
        df_val_test = df.iloc[idx_val_test, :].reset_index(drop=True)
        return ((X_train, Y_train, df.iloc[idx_train, :].reset_index(drop=True)), 
                (X_val, Y_val, df_val_test.iloc[idx_val, :].reset_index(drop=True)), 
                (X_test, Y_test, df_val_test.iloc[idx_test, :].reset_index(drop=True)))
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def subset_labels(Y, N_min=10, N_max=100, seed=None):
    """ Subset dataset to contain minimum N_min and maximum N_max instances 
        per class. Return indices for this subset. 
    
    Parameters
    ----------
    Y : np.array
        Class labels
    N_min : int
        Minimum instances per class
    N_max : int
        Maximum instances per class
    seed: int
        Random seed.
    
    Returns
    -------
    subset_idx : np.array
        Indices for a subset with classes of size bounded by N_min, N_max
    
    """
    np.random.seed(seed=seed)
    subset_idx = []
    labels = np.unique(Y)
    for label in labels:
        label_idx = np.where(Y==label)[0]
        counts = len(label_idx)
        if counts>=N_max:
            # too many samples from this class, keep only N_max
            subset_idx.append(np.random.choice(label_idx, N_max, replace=False))
        elif counts>=N_min and counts<N_max:
            # just enough samples, keep all of them
            subset_idx.append(label_idx)
        else:
            # not enough samples for this class, skip
            continue
    if len(subset_idx)>0:
        subset_idx = np.concatenate(subset_idx, axis=0)
    return subset_idx


def load_pretrained_model(basename):
    """Load model
    
    Parameters
    ----------
    basename : str
        Path to pretrained model saved in a .json and .h5 files.
       
    Returns
    -------
    loaded_model : Keras instance
        The model with pretrained weights.
    """
    # load json and create model
    json_file = open(basename+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(basename+".h5")
    print("Loaded model from disk")
    return loaded_model


def write_model(model, basename):
    # model to JSON
    model_json = model.to_json()
    with open(basename + ".json", "w") as json_file:
        json_file.write(model_json)
    
    # weights to HDF5
    model.save_weights(basename + ".h5")
    print("Saved model into h5 file")


def train_model(model, X_train, Y_train, X_val, Y_val, tags, batch_size=None, epochs=100):
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, Y_val))
    print(model.evaluate(X_val, Y_val))
    return model
 

def test_model(model, X_test, Y_test, batch_size=32):
    pred_tags = model.predict(X_test, batch_size=batch_size)
    auc = roc_auc_score(Y_test, pred_tags, average='micro')
    print(auc)
    return auc, pred_tags


def evaluate_model(model, X_test, Y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['accuracy', 'top_k_categorical_accuracy'])
    print(model.evaluate(X_test, Y_test))

