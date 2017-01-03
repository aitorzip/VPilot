#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from model import AitorNet
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
	datasetFiles = ['/home/aitor/Dataset/GTAVDataset_3/dataset.txt', '/home/aitor/Dataset/GTAVDataset_5/dataset.txt', '/home/aitor/Dataset/GTAVDataset_6/dataset.txt', 
					'/home/aitor/Dataset/GTAVDataset_7/dataset.txt', '/home/aitor/Dataset/GTAVDataset_8/dataset.txt', '/home/aitor/Dataset/GTAVDataset_3_2/dataset.txt', 
					'/home/aitor/Dataset/GTAVDataset_8_2/dataset.txt']
	lrcn = AitorNet()

	directories = []
	train_datasets = []
	val_datasets = []
	train_samples = 0
	val_samples = 0
	for datasetFile in datasetFiles:
		directories.append(os.path.dirname(datasetFile))
		dataset = lrcn.toSequenceDataset(datasetFile)

		val_len = int(len(dataset)*0.3)
		val_dataset = dataset[0:val_len]
		dataset = np.delete(dataset, np.s_[0:val_len], 0)
		
		train_samples = train_samples + len(dataset)
		val_samples = val_samples + len(val_dataset)

		train_datasets.append(dataset)
		val_datasets.append(val_dataset)	

	model = lrcn.getModel()
	model.compile(optimizer=Adam(), loss='mse')

	ckp_callback = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode='min')
	train_generator = lrcn.dataGenerator(directories, train_datasets)
	val_generator = lrcn.dataGenerator(directories, val_datasets)
	
	model.fit_generator(
		train_generator,
		samples_per_epoch=train_samples,
		nb_epoch=1000,
		validation_data=val_generator,
		nb_val_samples=val_samples,
		callbacks=[ckp_callback]
	)
	
