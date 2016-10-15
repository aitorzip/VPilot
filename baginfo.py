# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:38:10 2016

@author: aitor
"""
import utils

#bag = rosbag.Bag(datafile)
#print bag.get_message_count('/center_camera/image_color')
#print bag.get_message_count('/left_camera/image_color')
#print bag.get_message_count('/right_camera/image_color')

#utils.crop_rosbag_file("/media/aitor/Data/udacity/dataset.bag", "/media/aitor/Data/udacity/dataset-croped.bag", 1700)
utils.clean_rosbag_file("/media/aitor/Data/udacity/dataset1.bag", "/media/aitor/Data/udacity/dataset1-clean.bag")
utils.clean_rosbag_file("/media/aitor/Data/udacity/dataset2.bag", "/media/aitor/Data/udacity/dataset2-clean.bag")
utils.clean_rosbag_file("/media/aitor/Data/udacity/dataset3.bag", "/media/aitor/Data/udacity/dataset3-clean.bag")

