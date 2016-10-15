#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:56:59 2016

@author: aitor
"""

import nnmodel
import utils
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping                

# Train model--------------------
model = nnmodel.getNNModel(reg_lambda=1)
optimizer = Adam()
model.compile(optimizer=optimizer, loss="mse")
stopping_callback = EarlyStopping(patience=20)

train_generator = utils.udacity_data_generator(100)
val_data = utils.validation_udacity_data(1000)

model.fit_generator(
    train_generator,
    samples_per_epoch=100,
    nb_epoch=500,
    validation_data=val_data,
    nb_val_samples=100,
    callbacks=[stopping_callback]
)
#-----------------------------

#Save it if it is ok-----------
response = utils.query_yes_no("Training session has finished. Do you want to save the model?")
if response:
    model.save("/media/aitor/Data/udacity/model.h5")
#-----------------------------

#Show results-----------------
real_steering = 0
x = np.empty([1, 66, 200, 3])
for topic, msg, t in rosbag.Bag("/media/aitor/Data/udacity/dataset1-clean.bag").read_messages(topics=['/vehicle/steering_report', '/center_camera/image_color']):
	if(topic == '/vehicle/steering_report'):
		real_steering = msg.steering_wheel_angle
	elif(topic == '/center_camera/image_color'):
		x[0] = cv2.resize(CvBridge().imgmsg_to_cv2(msg, "bgr8"), (200, 66))
		y = model.predict(x, batch_size=1)
		print "real: " + str(real_steering) + ", predicted: " + str(y)
#----------------------------
