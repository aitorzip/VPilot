#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:22:21 2016

@author: aitor.
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

import numpy as np
from scipy.misc import imread, imresize
import csv
import cv2
import os

#Data generator to save memory resources
def data_generator(image_paths, steering_angles, batch_size, img_size=None):  
    while 1:
        img_list = []
        steering_angles_list = []
        i = 0
        for im_path in image_paths:
            img = imread(im_path)
            if img_size:
                img = imresize(img,img_size)
            img = img.astype('float32')
    
            img_list.append(img)
            steering_angles_list.append(steering_angles[i])
            
            #extra images (double), flip horizontally
            img_list.append(cv2.flip(img,1))
            steering_angles_list.append(-steering_angles[i])
            
            i = i+1
            
            if((i % batch_size) == 0):
                #shuffle the batch
                rng_state = np.random.get_state()
                np.random.shuffle(img_list)
                np.random.set_state(rng_state)
                np.random.shuffle(steering_angles_list)
                
                img_batch = np.stack(img_list, axis=0)
                steering_angles_batch = np.array(steering_angles_list)
                yield(img_batch, steering_angles_batch)
                img_list.clear()
                steering_angles_list.clear()  
                
                
            
    


#Based on:
#   75% NVIDIA's end-to-end paper: https://arxiv.org/pdf/1604.07316v1.pdf
#   25% Comma.ai research: https://github.com/commaai/research/blob/master/SelfSteering.md
def getNNModel(model_path=None):
      
    if model_path:
        with open(model_path, 'r') as jfile:
            model = model_from_json(jfile.read())
    else:

        ch, width, height = 3, 66, 200
        
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(height, width, ch), output_shape=(height, width, ch)))
        
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
    

#Load dataset
image_paths = []
steering_angles = []
with open('Center/driving_log.csv', 'r') as csvfile:
    dataset = csv.reader(csvfile)
    for row in dataset:
        image_paths.append(row[0].strip())
        image_paths.append(row[1].strip())
        image_paths.append(row[2].strip())
        steering_angles.append(float(row[3]))
        steering_angles.append(float(row[3]) + 0.2)
        steering_angles.append(float(row[3]) - 0.2)
        
with open('Right offset/driving_log.csv', 'r') as csvfile:
    dataset = csv.reader(csvfile)
    for row in dataset:
        if (float(row[3]) < 0.0): #only recoveries!
            image_paths.append(row[0].strip())
            image_paths.append(row[1].strip())
            image_paths.append(row[2].strip())
            steering_angles.append(float(row[3]))
            steering_angles.append(float(row[3]) + 0.2)
            steering_angles.append(float(row[3]) - 0.2)
            
with open('Left Offset/driving_log.csv', 'r') as csvfile:
    dataset = csv.reader(csvfile)
    for row in dataset:
        if (float(row[3]) > 0.0): #only recoveries!
            image_paths.append(row[0].strip())
            image_paths.append(row[1].strip())
            image_paths.append(row[2].strip())
            steering_angles.append(float(row[3]))
            steering_angles.append(float(row[3]) + 0.2)
            steering_angles.append(float(row[3]) - 0.2)


#shuffle the dataset
rng_state = np.random.get_state()
np.random.shuffle(image_paths)
np.random.set_state(rng_state)
np.random.shuffle(steering_angles)

#Split into training and validation, 80/20%
dataset_length = len(steering_angles)
train_image_paths = image_paths[0:int(dataset_length*0.8)]
train_steering_angles = steering_angles[0:int(dataset_length*0.8)]
val_image_paths = image_paths[int(dataset_length*0.8):dataset_length]
val_steering_angles = steering_angles[int(dataset_length*0.8):dataset_length]
    
#Training parameters                                    
adam = Adam(lr=0.001)
model = getNNModel(model_path="model.json")
if os.path.exists("model.h5"):
    model.load_weights("model.h5")
model.compile(optimizer=adam, loss='mse')

#Trains indefinetely and saves best model (lower validation loss)
checkpoint_callback = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
train_generator = data_generator(train_image_paths, train_steering_angles, 42, img_size=(200,66))
val_generator = data_generator(val_image_paths, val_steering_angles, 162, img_size=(200,66))
model.fit_generator(
                    train_generator, 
                    samples_per_epoch=2*len(train_steering_angles), #double, because we flip images horizontally in the generator
                    nb_epoch=1000,
                    validation_data=val_generator,
                    nb_val_samples=2*len(val_steering_angles), 
                    callbacks=[checkpoint_callback]
)
