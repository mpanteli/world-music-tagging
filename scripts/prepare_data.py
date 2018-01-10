# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 15:08:23 2017

@author: mariapanteli
"""
import numpy as np
import h5py
from sklearn.preprocessing import LabelBinarizer

import librosa

def read_dataset(hdf5_path, n_samples=None, i_start=None):
    '''Reads the dataset stored in hdf5_path and 
    returns numpy arrays of data X and labels Y.
    '''
    hdf5_file = h5py.File(hdf5_path, 'r')
    X = hdf5_file.get('X')
    Y = hdf5_file.get('Y')
    n = len(Y)
    if i_start is None:
        i_start = 0
    if n_samples is None:
        n_samples = n
    i_end = np.min([i_start + n_samples, n])
    X, Y = X[i_start:i_end], Y[i_start:i_end]
    hdf5_file.close()
    return X, Y


def encode_labels(labels, min_n_tags=50):
    '''Choose tags with a count of min_n_tags and convert 
    to binary vector using one-hot encoding. 
    
    Parameters
    ----------
    labels : np.array 2D or 1D
    '''
    binarizer = LabelBinarizer()
    tags = []
    one_hot = []
    for column in range(labels.shape[1]):
        labels_to_transform = labels[:, column]
        binarizer.fit(labels_to_transform)
        one_hot.append(binarizer.transform(labels_to_transform))
        tags.append(binarizer.classes_)
    tags = np.concatenate(tags)
    one_hot = np.concatenate(one_hot, axis=1)
    include_idx = select_n_tags(one_hot, tags, min_n_tags=min_n_tags)
    # return selected encoded_Y, tags
    if len(include_idx)==0:
        raise ValueError('No tags were found with min_n_tags occurrences. Consider decreasing min_n_tags or expanding the dataset.')
    one_hot = one_hot[:, include_idx]
    Y = one_hot
    tags = tags[include_idx]
    return Y, tags


def select_n_tags(encoded_Y, classes, min_n_tags=50):
    '''Choose tags with a count of min_n_tags.
    '''
    exclude_tags_idx = np.where(encoded_Y.sum(axis=0) < min_n_tags)[0]
    exclude_tag_nan = np.where(classes=='nan')[0]
    exclude_set_idx = set(exclude_tags_idx) | set(exclude_tag_nan)
    include_set_idx = list(set(range(len(classes))) - exclude_set_idx)
    include_idx = np.array(include_set_idx)
    return include_idx


def remove_samples_with_no_tags(X, Y, df=None):
    '''Given binary encoded labels Y remove observations X that 
    ended having a tag vector of zeros (i.e., the original tags of 
    these observations did not occur min_n_tag times in the dataset). 
    '''
    labels_idx = np.where(np.sum(Y, axis=1)>0)[0]
    X = X[labels_idx, :]
    Y = Y[labels_idx, :]
    if df is not None:
        df = df.iloc[labels_idx, :].reset_index(drop=True)
        return X, Y, df
    return X, Y


def mfcc_from_melspec(melspec, n_mfcc=20):
    """Extract MFCCs from given mel-spectrogram.

    Parameters
    ----------
    melspec : np.ndarray 
        Log-power mel spectrogram with dimensions freq x time

    Returns
    -------
    mfcc_feat : np.ndarray
        Mean and std of mfcc, delta_mmfcc, and delta_delta_mfcc 
    """
    mfcc = librosa.feature.mfcc(S=melspec, n_mfcc=n_mfcc+1)
    mfcc = mfcc[1:, :]  # ignore DC component - 1st coeff
    dmfcc = np.diff(mfcc, n=1, axis=1)
    ddmfcc = np.diff(mfcc, n=2, axis=1)
    mfcc_feat = np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1))
                          , axis=0)
    return mfcc_feat

