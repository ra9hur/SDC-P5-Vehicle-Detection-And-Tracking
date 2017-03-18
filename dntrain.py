#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:51:19 2017

@author: raghu
"""

import cv2
import glob
import numpy as np
import time
import matplotlib.image as mpimg

import dnmodel_def

from keras.preprocessing.image import load_img, img_to_array

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


img_width, img_height = 64, 64     # dimensions of image
nb_epoch = 20
BATCH_SIZE = 64

# -------------------- Pre-process / Augment data -----------------------------
# Flip image horizontally
def flip_image(image):

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()

    if flip_prob > 0.5:
        image = cv2.flip(image, 1)

    return image


def image_translate(image):
    
    rows,cols,ch = image.shape
    
    tx = int(10*np.random.uniform())
    ty = int(10*np.random.uniform())
    
    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(image, M, (cols,rows))

    return dst
    

def preprocess_augment(path, y, train):

    image = load_img(path)
    image = img_to_array(image)
    
    # Augment data applicable only during training
    if (train):
        image = flip_image(image)
        image = image_translate(image)
    return image, y


#   Ref: https://www.youtube.com/watch?v=bD05uGo_sVI&feature=youtu.be
def get_data_generator(X, y, batch_size, train=False):

    N = len(X)
    batches_per_epoch = N // batch_size

    # Shuffle rows after each epoch
    X, y = shuffle(X, y)

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 3, img_height, img_width), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        
        for a, b in zip(X[start:end],y[start:end]):
            X_batch[j], y_batch[j] = preprocess_augment(a, b, train)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index to zero
            i = 0
        yield X_batch, y_batch

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":

# -------------------- Load / Prepare data ------------------------------------
    
    # Load images
    car_images = glob.glob("./vehicles/*/*/*.png")
    notcar_images = glob.glob("./non-vehicles/*/*/*.png")
    X = []
    y = []
    cars = []
    notcars = []

    for image in car_images:
        cars.append(image)

    for image in notcar_images:
        notcars.append(image)

    # Prepare data and the corresponding label vectors
    X.extend(cars)
    X.extend(notcars)

    # Define the labels vector
    y.extend(np.ones(len(cars)))
    y.extend(np.zeros(len(notcars)))


# -------------------- Split data to train / valid / test datasets ------------
    # Split up data into randomized training and test sets
    X, y = shuffle(X, y)
    rand_state = np.random.randint(0, 100)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)

    X_tv, y_tv = shuffle(X_tv, y_tv)
    rand_state = np.random.randint(0, 100)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    ntrain = len(X_train)
    nvalid = len(X_valid)


# -------------------- Generators to load data in batches ---------------------
    train_generator = get_data_generator(X_train, y_train, batch_size=BATCH_SIZE, train=True)

    validation_generator = get_data_generator(X_valid, y_valid, batch_size=BATCH_SIZE)


# -------------------- Define and train the model -----------------------------
    sdc_model = dnmodel_def.custom_model()


    nb_train_samples = (ntrain//BATCH_SIZE)*BATCH_SIZE
    nb_validation_samples = (nvalid//BATCH_SIZE)*BATCH_SIZE

    t1 = time.time()

    sdc_model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# -------------------- Test and save the model --------------------------------
    t2 = time.time()
    print("Time taken to train the model: ",round(t2-t1, 3))

    test_size = len(X_test)

    X_tst = np.zeros((test_size, 3, img_height, img_width), dtype=np.float32)
    y_tst = np.zeros((test_size,), dtype=np.float32)

    i = 0
    for path, y_temp in zip(X_test,y_test):
        x_temp = load_img(path)
        X_tst[i] = img_to_array(x_temp)
        y_tst[i] = y_temp
        i += 1

    # Check the score of the model
    test_loss, test_acc = sdc_model.evaluate(X_tst, y_tst)
    print('Test loss / accuracy = ', round(test_loss,3), round(test_acc,3))
    print('Test Accuracy of the model = ', round(test_acc, 3))

    # Check the prediction time for a single sample
    n_predict = 10

    ret = sdc_model.predict(X_tst[0:n_predict])
    raw = [ '%.2f' % elem for elem in ret ]
    results = []

    for i in ret:
        if i > 0.5: results.append(1.0)
        else: results.append(0.0)
    
    print('My model predicts (raw): ', np.transpose(raw))
    print('My model predicts: ', np.transpose(results))
    print('For ',n_predict, 'labels: ', y_tst[0:n_predict])
    t3 = time.time()
    print(round(t3-t2, 3), 'Seconds to predict', n_predict,'labels with the model') 
    

    # SAVE WEIGHTS - save the weights of a model
    sdc_model.save_weights('model_folder/model.h5')

    
