#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:01:55 2016

@author: aitor
"""

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

#Based on:
#   75% NVIDIA's end-to-end paper: https://arxiv.org/pdf/1604.07316v1.pdf
#   25% Comma.ai research: https://github.com/commaai/research/blob/master/SelfSteering.md
def getNNModel(model_path=None, reg_lambda=0.0):
      
    if model_path:
        model = load_model(model_path)
    else:

        ch, width, height = 3, 200, 66
        
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(height, width, ch), output_shape=(height, width, ch)))
    	
        model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='same', W_regularizer=l2(reg_lambda), name='conv1'))
        model.add(ELU())
        #model.add(MaxPooling2D((2,2)))
    
        model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='same', W_regularizer=l2(reg_lambda), name='conv2'))
        model.add(ELU())
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
        #model.add(MaxPooling2D((2,2)))
    
        model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', W_regularizer=l2(reg_lambda), name='conv3'))
        model.add(ELU())
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
        #model.add(MaxPooling2D((2,2)))
    
        model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(reg_lambda), name='conv4'))
        model.add(ELU())
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
        #model.add(MaxPooling2D((2,2)))
    
        model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(reg_lambda), name='conv5'))
        model.add(ELU())
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
        #model.add(MaxPooling2D((2,2)))
        
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(ELU())
         
        model.add(Dense(100, W_regularizer=l2(reg_lambda), name='dense_1'))
        model.add(Dropout(0.5))
        model.add(ELU())
        model.add(Dense(50, W_regularizer=l2(reg_lambda), name='dense_2'))
        model.add(Dropout(0.5))
        model.add(ELU())
        model.add(Dense(10, W_regularizer=l2(reg_lambda), name='dense_3'))
        model.add(Dropout(0.5))
        model.add(ELU())
        model.add(Dense(1, W_regularizer=l2(reg_lambda), name='output'))
    
    return model
