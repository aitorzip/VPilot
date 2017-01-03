#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imresize
import os
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

#   Basically a modified AlexNet (based on Nvidia's arch) with an LSTM stacked on top of it.
class AitorNet:
	def __init__(self, lookback=50, width=320, height=160, channels=3):
		self.lookback = lookback
		self.width = width
		self.height = height
		self.channels = channels

	def getModel(self, weights_path=None):
		model = Sequential()
		model.add(TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), input_shape=(self.lookback, self.height, self.width, self.channels)))

		model.add(TimeDistributed(Convolution2D(96, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv1')))
		model.add(ELU())
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(144, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv2')))
		model.add(ELU())
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(192, 5, 5, subsample=(2,2), border_mode='same', init='he_normal', name='conv3')))
		model.add(ELU())
		model.add(BatchNormalization())

		model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', name='conv4')))
		model.add(ELU())
		model.add(BatchNormalization())
		
		model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', name='conv5')))
		model.add(ELU())
		model.add(BatchNormalization())

		model.add(TimeDistributed(Flatten()))

		model.add(TimeDistributed(Dense(512, init='he_normal', name='Dense_1')))
		model.add(ELU())
		model.add(Dropout(0.5))

		model.add(LSTM(256, return_sequences=False, init='he_normal', name='LSTM_1'))
		#model.add(ELU())
		model.add(Dropout(0.5))

		model.add(Dense(3, init='he_normal', name='output'))	

		if(weights_path):
			model.load_weights(weights_path)

		return model
		
	
	def toSequenceDataset(self, datafile, shuffle=True):
		originalDataset = np.genfromtxt(datafile, delimiter=' ')
		nframes = originalDataset.shape[0]
		ntargets = originalDataset.shape[1]

		sequenceDataset = []
		for frameIndex in range(1, nframes+1):
			x = np.zeros(self.lookback, dtype='int32')
			if((frameIndex - self.lookback) < 0):
				x[-frameIndex:] = originalDataset[:frameIndex, 0]
			else:
				x[:] = originalDataset[(frameIndex - self.lookback):frameIndex, 0]

			sequence = (x, originalDataset[frameIndex-1, 1:ntargets])
			sequenceDataset.append(sequence)

		if(shuffle):
			random.shuffle(sequenceDataset)

		return sequenceDataset
	
	def dataGenerator(self, directories, datasets, batchSize=1):
		while True:
			x = np.zeros((batchSize, self.lookback, self.height, self.width, self.channels), dtype='float32')
			y = np.zeros((batchSize, 3))
			batchCount = 0

			for directory, dataset in zip(directories, datasets):
				for sample in dataset:
					y[batchCount, :] = sample[1][2:5]
					for frameIndex, frameName in enumerate(sample[0]):
						if(int(frameName) == 0):
							pass
						else:
							x[batchCount, frameIndex, :, :, :] = imread(os.path.join(directory, str(int(frameName))) + ".png", mode="RGB").astype('float32')
							
					batchCount = batchCount + 1
					if (batchCount == batchSize):
						yield(x, y)
						batchCount = 0
						x.fill(0)
						y.fill(0) 

   
