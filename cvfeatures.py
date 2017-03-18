#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 06:32:34 2017

@author: raghu
"""

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
#from sklearn.decomposition import PCA


# ---------- Includes functions to extract features ---------------------------


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Flip image horizontally
def flip_image(image):

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()

    if flip_prob > 0.5:
        image = cv2.flip(image, 1)

    return image


# brightness adjustments
def adjust_brightness(image):

    # convert to HSV so that its easy to adjust brightness
    image2 = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image2[:,:,2] = image2[:,:,0]*random_bright

    # convert to RBG again
    image2 = cv2.cvtColor(image2,cv2.COLOR_YCrCb2RGB)

    return image2


def image_translate(image):
    
    rows,cols,ch = image.shape
    
    tx = int(10*np.random.uniform())
    ty = int(10*np.random.uniform())
    
    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(image, M, (cols,rows))

    return dst
    

# ---------- Spatial Binning of Color: color ----------------------------------
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size)
    features = features.ravel()
    
    # Return the feature vector
    return features


# ---------- Extract PCA as a feature -----------------------------------------
# http://stackoverflow.com/questions/35218995/image-classification-using-svm-python/35219151

def pca(X,color):

    # get dimensions
    n_components = 3

    if (color=='RGB'):
        X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    else:
        X = cv2.cvtColor(X, cv2.COLOR_YCrCb2RGB)
        X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)

    # PCA - SVD used
    U,S,V = np.linalg.svd(X)
    V = V[:n_components] # only makes sense to return the first num_data
    arr= np.array(V)
    pca_features = arr.ravel()
    # return the projection matrix, the variance and the mean
    return pca_features


# ---------- Histograms of pixel intensity (color histograms): color ----------
# Define a function to compute color histogram features  
def color_hist(img, nbins=32):      # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


# ---------- To achieve color invariance: shape -------------------------------
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,cspace='RGB',spatial_size=(32, 32),hist_bins=32, orient=9, pix_per_cell=8, 
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)

        image = flip_image(image)
        #image = image_translate(image)
        #print("Image min-max: ",np.min(image), np.max(image))
        #image = image.astype(np.float32)/255

        feature = single_feature(image, cspace, spatial_size,hist_bins, orient, pix_per_cell, 
                            cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

        features.append(feature)

    # Return list of feature vectors
    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features(), just for a single image rather than list of images
def single_feature(img, color_space='RGB', spatial_size=(32, 32),hist_bins=32, orient=9, pix_per_cell=8, 
                        cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):    

    #1) Define an empty list to receive features
    img_features = []
    
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      


    #3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)


    #4) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)


    #pca_features = pca(img)
    #img_features.append(pca_features)

    
    #5) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, 
                                    cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, 
                                    cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)


    #6) Return concatenated array of features
    return np.concatenate(img_features)

