#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

class BaseModel:
	def __init__(self, lookback=50, width=320, height=160, channels=3):
		self.lookback = lookback
		self.width = width
		self.height = height
		self.channels = channels

	def toSequenceDataset(self, datafiles, shuffle=True):
		sequenceDataset = []

		for datafile in datafiles:
			originalDataset = np.genfromtxt(datafile, delimiter=' ')
			directory = os.path.dirname(datafile)
			filenames = [os.path.join(directory, str(int(imageName))) + '.png' for imageName in originalDataset[:, 0]]

			nframes = originalDataset.shape[0]
			ntargets = originalDataset.shape[1]
			for frameIndex in range(1, nframes+1):
				x = [None] * self.lookback
				if((frameIndex - self.lookback) < 0):
					x[-frameIndex:] = filenames[:frameIndex]
				else:
					x[:] = filenames[(frameIndex - self.lookback):frameIndex]

				sequence = (x, originalDataset[frameIndex-1, 1:ntargets])
				sequenceDataset.append(sequence)

		if(shuffle):
			random.shuffle(sequenceDataset)

		return sequenceDataset
	
	def dataGenerator(self, dataset, mode=1, batchSize=1):
		while True:
			x = np.zeros((batchSize, self.lookback, self.height, self.width, self.channels), dtype='float32')
			y = np.zeros((batchSize, 3))
			batchCount = 0

			for sample in dataset:
				y[batchCount, :] = sample[1][2:5]
				for frameIndex, frameFile in enumerate(sample[0]):
					if(frameFile == None):
						pass
					else:
						x[batchCount, frameIndex, :, :, :] = cv2.imread(frameFile, mode).astype('float32')
							
				batchCount = batchCount + 1
				if (batchCount == batchSize):
					yield(x, y)
					batchCount = 0
					x.fill(0)
					y.fill(0) 


class nanoAitorNet(BaseModel):
	
	def getModel(self, weights_path=None):
		model = Sequential()
		model.add(TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), input_shape=(self.lookback, self.height, self.width, self.channels)))

		model.add(TimeDistributed(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv1')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv2')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv3')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='elu', name='conv4')))
		model.add(BatchNormalization())
		
		model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='elu', name='conv5')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Flatten()))

		model.add(TimeDistributed(Dense(1024, init='he_normal', activation='elu', name='Dense_1')))
		model.add(Dropout(0.5))

		model.add(LSTM(512, return_sequences=False, init='he_normal', name='LSTM_1'))
		model.add(Dropout(0.5))

		model.add(Dense(256, init='he_normal', activation='elu', name='Dense_2'))
		model.add(Dropout(0.5))

		model.add(Dense(3, init='he_normal', name='output'))	

		if(weights_path):
			model.load_weights(weights_path)

		return model

class AitorNet(BaseModel):
	
	def getModel(self, weights_path=None):
		model = Sequential()
		model.add(TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), input_shape=(self.lookback, self.height, self.width, self.channels)))

		model.add(TimeDistributed(Convolution2D(32, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv1')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv2')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(64, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', activation='elu', name='conv3')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(86, 3, 3, border_mode='same', init='he_normal', activation='elu', name='conv4')))
		model.add(BatchNormalization())
		
		model.add(TimeDistributed(Convolution2D(86, 3, 3, border_mode='same', init='he_normal', activation='elu', name='conv5')))
		model.add(BatchNormalization())

		model.add(TimeDistributed(Flatten()))

		model.add(TimeDistributed(Dense(1024, init='he_normal', activation='elu', name='Dense_1')))
		model.add(Dropout(0.5))

		model.add(LSTM(512, return_sequences=False, init='he_normal', name='LSTM_1'))
		model.add(Dropout(0.5))

		model.add(Dense(256, init='he_normal', activation='elu', name='Dense_2'))
		model.add(Dropout(0.5))

		model.add(Dense(3, init='he_normal', name='output'))	

		if(weights_path):
			model.load_weights(weights_path)

		return model
		
	
	

