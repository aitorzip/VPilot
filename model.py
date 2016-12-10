#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Lambda

#Based on:
#   75% NVIDIA's end-to-end paper: https://arxiv.org/pdf/1604.07316v1.pdf
#   25% Comma.ai research: https://github.com/commaai/research/blob/master/SelfSteering.md
def getModel(model_path=None):
      
    if model_path:
        with open(model_path, 'r') as jfile:
            model = model_from_json(jfile.read())
            weights_path = model_path.replace('json', 'h5')
            model.load_weights(weights_path)
    else:
        width, height, ch = 200, 66, 3
        
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(width, height, ch), output_shape=(width, height, ch)))
        
        model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv1'))
        model.add(ELU())
        model.add(BatchNormalization())
    
        model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv2'))
        model.add(ELU())
        model.add(BatchNormalization())
    
        model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv3'))
        model.add(ELU())
        model.add(BatchNormalization())
    
        model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', name='conv4'))
        model.add(ELU())
        model.add(BatchNormalization())
       
        model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', name='conv5'))
        model.add(ELU())
        model.add(BatchNormalization())
        
        model.add(Flatten())
         
        model.add(Dense(100, init='he_normal', name='dense_1'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(50, init='he_normal', name='dense_2'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(10, init='he_normal', name='dense_3'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(1, init='he_normal', name='output'))

    return model