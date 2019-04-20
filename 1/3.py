#This file creates .npy files for line data from the classes folder in q1_data 
#so that model can be trained for it
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

def read_data():
    path = os.path.dirname(os.path.realpath(__file__))+'/classes/'
    classes = ([cls for cls in os.listdir(path) if cls.startswith("class")])
    classes = os.listdir(path)
    print (len(classes))
    x_train=[]
    y_train=[]
    y_train_length=[]
    y_train_width=[]
    y_train_angle=[]
    y_train_color=[]
    x_test=[]
    y_test=[]
    y_test_length=[]
    y_test_width=[]
    y_test_angle=[]
    y_test_color=[]
    cnt=0
    # return
    for cls in classes:
        print (cls) 
        params = cls.split('_')
        print(params)
        length=int(params[0])
        width=int(params[1])
        angle = int(params[2])
        color=int(params[3])
        print(length,' ',width,' ',angle,' ',color)
        images = os.listdir(os.path.join(path,cls))
        print (len(images))
        train = int(0.6*len(images))
        test = int(0.4*len(images))
        print (train,test)
        i=0
        for image in images:
            img_path = os.path.join(path,cls)
            img_path = os.path.join(img_path,image)
            if(i<train):
                x_train.append(cv2.imread(img_path))
                y_train.append(cnt)
                y_train_length.append(length)
                y_train_width.append(width)
                y_train_angle.append(angle)
                y_train_color.append(color)
            elif(i-train<test):
                x_test.append(cv2.imread(img_path))
                y_test.append(cnt)
                y_test_length.append(length)
                y_test_width.append(width)
                y_test_angle.append(angle)
                y_test_color.append(color)
            i+=1
        cnt+=1

    print(y_train_length[52])
    print(y_train_angle[52])
    x_train=np.asarray(x_train)
    print (x_train.shape)
    y_train=np.asarray(y_train)
    print (y_train.shape)
    y_train_length=np.asarray(y_train_length)
    print (y_train_length.shape)
    y_train_width=np.asarray(y_train_width)
    print (y_train_width.shape)
    y_train_angle=np.asarray(y_train_angle)
    print (y_train_angle.shape)
    y_train_color=np.asarray(y_train_color)
    print (y_train_color.shape)
    
    
    x_test=np.asarray(x_test)
    print (x_test.shape)
    y_test=np.asarray(y_test)
    print (y_test.shape)
    y_test_length=np.asarray(y_test_length)
    print (y_test_length.shape)
    y_test_width=np.asarray(y_test_width)
    print (y_test_width.shape)
    y_test_angle=np.asarray(y_test_angle)
    print (y_test_angle.shape)
    y_test_color=np.asarray(y_test_color)
    print (y_test_color.shape)

    # for i in range(len(y_test_length)):
    #     print(y_test_length[i])

    np.save('q1_data/x_train', x_train)
    np.save('q1_data/y_train', y_train)
    np.save('q1_data/y_train_length', y_train_length)
    np.save('q1_data/y_train_width', y_train_width)
    np.save('q1_data/y_train_angle', y_train_angle)
    np.save('q1_data/y_train_color', y_train_color)

    np.save('q1_data/x_test', x_test)
    np.save('q1_data/y_test', y_test)
    np.save('q1_data/y_test_length', y_test_length)
    np.save('q1_data/y_test_width', y_test_width)
    np.save('q1_data/y_test_angle', y_test_angle)
    np.save('q1_data/y_test_color', y_test_color)


    # return x_train, y_train, x_test, y_test

read_data()
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
