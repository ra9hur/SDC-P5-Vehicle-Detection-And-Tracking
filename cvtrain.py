#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:06:59 2017

@author: raghu
"""

# import required packages
import numpy as np
import time
import pickle
import glob
from sklearn.svm import LinearSVC
#from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cvfeatures
import cvconfig

# Parameters from the config file
color_space = cvconfig.color_space        # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = cvconfig.orient                  # HOG orientations
pix_per_cell = cvconfig.pix_per_cell      # HOG pixels per cell
cell_per_block = cvconfig.cell_per_block  # HOG cells per block
spatial_size = cvconfig.spatial_size      # Spatial binning dimensions
hist_bins = cvconfig.hist_bins            # Number of histogram bins
hog_channel = cvconfig.hog_channel        # Can be 0, 1, 2, or "ALL"
spatial_feat = cvconfig.spatial_feat      # Spatial features on or off
hist_feat = cvconfig.hist_feat            # Histogram features on or off
hog_feat = cvconfig.hog_feat              # HOG features on or off


# Function to train the classifier
def process(cars, notcars):
    
    t=time.time()

    # Extract car features - binned spacial, color histogram and hog features
    car_features = cvfeatures.extract_features(cars, cspace=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel)

    # Extract non-car features - binned spacial, color histogram and hog features
    notcar_features = cvfeatures.extract_features(notcars, cspace=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    scaled_X, y = shuffle(scaled_X, y)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:',  len(X_train[0]))

    # Use a linear SVC 
    svc = LinearSVC()
    #svc = SGDClassifier()
    # Check the training time for the SVC
    t=time.time()
    X_train, y_train = shuffle(X_train, y_train)
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10

    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For ',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hist_bins"] = hist_bins
    dist_pickle["feature_length"] = len(X_train[0])
    pickle.dump( dist_pickle, open( "model_folder/svc_pickle.p", "wb" ) )



if __name__ == "__main__":

    # Load images
    car_images = glob.glob("./vehicles/*/*/*.png")
    notcar_images = glob.glob("./non-vehicles/*/*/*.png")
    cars = []
    notcars = []

    for image in car_images:
        cars.append(image)

    for image in notcar_images:
        notcars.append(image)

    process(cars, notcars)
