import os.path
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import sys
import h5py
import glob

def get_datafile():
    datafile = "dataset.bag"
    if os.path.exists("./data/" + datafile):
        datasetsDir = "./data/"
    elif os.path.exists("../data/" + datafile):
        datasetsDir = "../data/"
    else:
        datasetsDir = "/media/aitor/Data1/"
    return datasetsDir + datafile
    
def validation_udacity_data(batchsize, path="/media/aitor/Data/udacity/dataset2-clean.bag"):
    bag = rosbag.Bag(path)
    x = np.empty([batchsize, 66, 200, 3])
    y = np.empty([batchsize, 1])
    cvbridge = CvBridge()
    
    i = 0;
    current_steering = 0
    for topic,msg,t in bag.read_messages(topics=['/vehicle/steering_report', '/center_camera/image_color', '/center_camera/image_color/compressed']):
        if(topic == '/vehicle/steering_report'):
            current_steering = msg.steering_wheel_angle
        elif(topic == '/center_camera/image_color'):
            x[i] = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
            y[i] = np.array([current_steering]);
            i = i + 1
        elif(topic == '/center_camera/image_color/compressed'):
            x[i] = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
            y[i] = np.array([current_steering]);
            i = i + 1

        if(i == batchsize):
            i = 0
            return (x,y)

    
def udacity_data_generator(batchsize, path="/media/aitor/Data/udacity/dataset3-clean.bag", shift=None):
    cvbridge = CvBridge()
    #Not shited sequential data generator
    if (shift is None):
        while 1:
            bag = rosbag.Bag(path)
            x = np.empty([batchsize, 66, 200, 3])
            y = np.empty([batchsize, 1])
    
            i = 0;
            current_steering = 0
            for topic,msg,t in bag.read_messages(topics=['/vehicle/steering_report', '/center_camera/image_color', '/center_camera/image_color/compressed']):
                if(topic == '/vehicle/steering_report'):
                    current_steering = msg.steering_wheel_angle
                elif(topic == '/center_camera/image_color'):
                    x[i] = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1
                elif(topic == '/center_camera/image_color/compressed'):
                    x[i] = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1

                if(i == batchsize):
                    i = 0
                    yield (x,y)

            bag.close()
    else:
        #Shifted sequential data generator
        while 1:
            bag = rosbag.Bag(path)
            x = np.empty([batchsize, 66, 200, 3])
            y = np.empty([batchsize, 1])

            i = 0
            current_steering = 0;
            for topic,msg,t in bag.read_messages(topics=['/vehicle/steering_report', '/center_camera/image_color',  '/left_camera/image_color', '/right_camera/image_color', '/center_camera/image_color/compressed',  '/left_camera/image_color/compressed', '/right_camera/image_color/compressed']):
                if(topic == '/vehicle/steering_report'):
                    current_steering = msg.steering_wheel_angle
                elif(topic == '/center_camera/image_color'):
                    x[i] = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1
                elif(topic == '/center_camera/image_color/compressed'):
                    x[i] = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1
                elif(topic == '/left_camera/image_color'):
                    x[i] = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering + shift]);
                    i = i + 1
                elif(topic == '/left_camera/image_color/compressed'):
                    x[i] = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1
                elif(topic == '/right_camera/image_color'):
                    x[i] = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering - shift]);
                    i = i + 1
                elif(topic == '/right_camera/image_color/compressed'):
                    x[i] = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                    y[i] = np.array([current_steering]);
                    i = i + 1
                if(i == batchsize):
                    i = 0
                    yield (x, y)
        
            bag.close()

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
        

def clean_rosbag_file(inpath, outpath):
	with rosbag.Bag(outpath, 'w') as outbag:
         current_speed = 0
         current_steering = 0
         for topic, msg, t in rosbag.Bag(inpath).read_messages(topics=['/vehicle/steering_report', '/center_camera/image_color', '/center_camera/image_color/compressed']):
            if (topic == '/vehicle/steering_report'):
                current_speed = msg.speed
                current_steering = msg.steering_wheel_angle

            if ((current_speed > 8.0) and (abs(current_steering) >= 0.1)):
                outbag.write(topic, msg, t)


def load_deepdrive_files(filesdir):
    #each file is considered as a batch of data
    dfiles = glob.glob(filesdir + "/*.h5")
    for dfile in dfiles:
        with h5py.File(dfile, 'r') as h5f:
            data = dict(h5f.items())
            #convert from float32 to uint8
            images = np.array(data['images'].value, dtype=np.uint8)
            targets = np.array(data['targets'].value)
            #vehicle_states = np.array(data['vehicle_states'].value)
            #clear the data to save memory
            # data = None
            yield images, targets
