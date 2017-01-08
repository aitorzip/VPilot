#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model import nanoAitorNet

if __name__ == '__main__':

	datasetFiles = ['/media/aitor/Data/GTAVDataset_3/dataset.txt', '/media/aitor/Data/GTAVDataset_5/dataset.txt', '/media/aitor/Data/GTAVDataset_6/dataset.txt', 
					'/media/aitor/Data/GTAVDataset_7/dataset.txt', '/media/aitor/Data/GTAVDataset_8/dataset.txt', '/media/aitor/Data/GTAVDataset_3_2/dataset.txt', 
					'/media/aitor/Data/GTAVDataset_8_2/dataset.txt']
	
	aitorNet = nanoAitorNet()

	dataset = aitorNet.toSequenceDataset(datasetFiles)
	valLen = int(len(dataset)*0.3)
	valDataset = dataset[0:valLen]
	dataset = np.delete(dataset, np.s_[0:valLen], 0)

	trainGenerator = aitorNet.dataGenerator(dataset)
	valGenerator = aitorNet.dataGenerator(valDataset)

	model = aitorNet.getModel()
	model.compile(optimizer=Adam(clipnorm=1.0), loss='mse')
	ckp_callback = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode='min')
	
	model.fit_generator(
		trainGenerator,
		samples_per_epoch=len(dataset),
		nb_epoch=1000,
		validation_data=valGenerator,
		nb_val_samples=len(valDataset),
		callbacks=[ckp_callback]
	)
