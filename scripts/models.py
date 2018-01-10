# -*- coding: utf-8 -*-
"""
Created on Fri Oct 08 10:18:10 2017

@author: mariapanteli
"""
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalMaxPooling2D


def cnn_2L(input_shape, nb_classes):
    """CNN with 2 layers
    """
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))

    model.add(Conv2D(64, (5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dense(nb_classes, activation='sigmoid'))

    # print summary
    model.summary()

    return model


def cnn_4L(input_shape, nb_classes):
    """CNN with 4 layers
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1, 621), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Dense(nb_classes, activation='sigmoid'))

    # print model summary
    model.summary()

    return model


def mfcc_nn(input_shape, nb_classes):
    """Tagging with mfcc features using a neural network 
        with one dense layer (baseline model).
    """
    model = Sequential()
    model.add(Dense(nb_classes, input_shape=input_shape, activation='sigmoid'))
    model.summary()
    return model