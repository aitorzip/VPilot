#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Dataset, frame2numpy, Scenario
from deepgtav.client import Client

import argparse
import time
import cv2

# Stores a dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default='dataset.pz', help='Place to store the dataset')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port. 
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)
    
    # Configures the information that we want DeepGTAV to generate and send to us. 
    # See deepgtav/messages.py to see what options are supported
    dataset = Dataset(rate=30, frame=[320,160], throttle=True, brake=True, steering=True, vehicles=True, peds=True, reward=[15.0, 0.0], direction=None, speed=True, yawRate=True, location=True, time=True)
    # Send the Start request to DeepGTAV.
    scenario = Scenario(drivingMode=[786603,15.0]) # Driving style is set to normal, with a speed of 15.0 mph. All other scenario options are random.
    client.sendMessage(Start(dataset=dataset,scenario=scenario))

    # Start listening for messages coming from DeepGTAV. We do it for 80 hours
    stoptime = time.time() + 80*3600
    while time.time() < stoptime:
        try:
            # We receive a message as a Python dictionary
            message = client.recvMessage()  
            
            # The frame is a numpy array and can be displayed using OpenCV or similar       
            # image = frame2numpy(message['frame'], (320,160))
            # cv2.imshow('img',image)
            # cv2.waitKey(-1)
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()
