#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:10:30 2017

@author: raghu
"""


# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


from keras.models import Sequential
from keras.layers.core import Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU, Dropout
import numpy as np

# dimensions of our images.
batch_size = 64


# https://github.com/fchollet/keras/issues/1920
#def yolo_model(input_shape=(3,64,64), weights='model_folder/yolo-tiny.weights'):
# 'yolo-tiny.weights' is not included in github to avoid uploading 180MB file
def yolo_model(input_shape=(3,64,64), weights='model_folder/yolo_extract.h5'):

    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0, input_shape=input_shape, output_shape=input_shape))

    model.add(Convolution2D(16, 3, 3, border_mode='same',subsample=(1,1), trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32,3,3 ,border_mode='same', trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    model.add(Convolution2D(64,3,3 ,border_mode='same', trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

    model.add(Convolution2D(128,3,3 ,border_mode='same', trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

    model.add(Convolution2D(256,3,3 ,border_mode='same', trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

    model.add(Convolution2D(512,3,3 ,border_mode='same', trainable=False))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

    '''
    #model.add(Convolution2D(1024,3,3 ,border_mode='same', trainable=False))
    #model.add(LeakyReLU(alpha=0.1))

    #model.add(Convolution2D(1024,3,3 ,border_mode='same', trainable=False))
    #model.add(LeakyReLU(alpha=0.1))

    #model.add(Convolution2D(1024,3,3 ,border_mode='same', trainable=False))
    #model.add(LeakyReLU(alpha=0.1))


    if weights:
        data = np.fromfile(weights,np.float32)
        data=data[4:]
    
        index = 0
        for layer in model.layers:
            shape = [w.shape for w in layer.get_weights()]
            if shape != []:
                kshape,bshape = shape
                bia = data[index:index+np.prod(bshape)].reshape(bshape)
                index += np.prod(bshape)
                ker = data[index:index+np.prod(kshape)].reshape(kshape)
                index += np.prod(kshape)
                layer.set_weights([ker,bia])

    model.save_weights('model_folder/yolo_extract.h5')
    '''

    if weights:
        model.load_weights(weights)

    model.add(Convolution2D(10, 3, 3, activation='relu', border_mode="same"))
    model.add(Convolution2D(10, 3, 3, activation='relu', border_mode="same"))
    #model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(128, 8, 8, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,1,1, activation="sigmoid"))

    return model

    
def custom_model(input_shape=(3,64,64)):

    model = yolo_model(input_shape=(3,64,64))
    model.add(Flatten())
    
    model.compile('adadelta', 'mse', ['accuracy'])
    #model.compile('adadelta', 'binary_crossentropy', ['accuracy'])

    return model


if __name__ == "__main__":

    model = custom_model()
    model.summary()
