#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client

import argparse
import time
import cv2

class Model:
    def run(self,frame):
        return [1.0, 0.0, 0.0] # throttle, brake, steering

# Controls the DeepGTAV vehicle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port. 
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    # We don't want to save a dataset in this case
    client = Client(ip=args.host, port=args.port)
    
    # We set the scenario to be in manual driving, and everything else random (time, weather and location). 
    # See deepgtav/messages.py to see what options are supported
    scenario = Scenario(drivingMode=-1) #manual driving
    
    # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
    client.sendMessage(Start(scenario=scenario))
    
    # Dummy agent
    model = Model()

    # Start listening for messages coming from DeepGTAV. We do it for 80 hours
    stoptime = time.time() + 80*3600
    while time.time() < stoptime:
        try:
            # We receive a message as a Python dictionary
            message = client.recvMessage()  
                
            # The frame is a numpy array that can we pass through a CNN for example     
            image = frame2numpy(message['frame'], (320,160))
            commands = model.run(image)
            # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
            client.sendMessage(Commands(commands[0], commands[1], commands[2]))
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()
