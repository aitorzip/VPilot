#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from model import nanoAitorNet

def acceptSample(sample):
	throttle = sample[1][4]
	steering = sample[1][3]

	if (np.absolute(steering) > 0.025 and throttle > 0.15):
		return True
	
	if(throttle < 0.15 or np.absolute(steering) < 0.025):
		if(np.random.rand() > 0.95):
			return True

	return False

def plotHist(dataset, n):
	values = [sample[1][n]for sample in dataset]

	plt.hist(values, bins='auto')
	plt.show()

if __name__ == '__main__':

	datasetFiles = ['/media/aitor/Data/GTAVDataset_3/dataset.txt', '/media/aitor/Data/GTAVDataset_5/dataset.txt', '/media/aitor/Data/GTAVDataset_6/dataset.txt', 
					'/media/aitor/Data/GTAVDataset_7/dataset.txt', '/media/aitor/Data/GTAVDataset_8/dataset.txt', '/media/aitor/Data/GTAVDataset_3_2/dataset.txt', 
					'/media/aitor/Data/GTAVDataset_8_2/dataset.txt']
	
	aitorNet = nanoAitorNet()

	dataset = aitorNet.toSequenceDataset(datasetFiles)	
	dataset = [sample for sample in dataset if acceptSample(sample)]
	
	valLen = int(len(dataset)*0.2)
	valDataset = dataset[0:valLen]
	dataset = np.delete(dataset, np.s_[0:valLen], 0)

	trainGenerator = aitorNet.dataGenerator(dataset)
	valGenerator = aitorNet.dataGenerator(valDataset)

	model = aitorNet.getModel(weights_path="model.h5")
	model.compile(optimizer=RMSprop(), loss='mse')
	ckp_callback = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode='min')
	
	model.fit_generator(
		trainGenerator,
		samples_per_epoch=len(dataset),
		nb_epoch=1000,
		validation_data=valGenerator,
		nb_val_samples=len(valDataset),
		callbacks=[ckp_callback]
	)
