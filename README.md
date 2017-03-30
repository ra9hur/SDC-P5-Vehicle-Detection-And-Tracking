# SDC-P5-Vehicle-Detection-And-Tracking

----------
**1. Problem Definition - About Project**
-------------
In self-driving cars, it is important to know, where other vehicles are on the road and be able to anticipate where they're heading next. Essentially, it is required to determine things like how far away they are, which way they're going, and how fast they're moving.

The goal of this project is to use computer vision techniques to,

 - Explore visual features that can be extracted from images in order to reliably classify vehicles 
 - Search an image for vehicles, separate false positives from real detection 
 - Apply similar ideas and track those detections from frame to frame in a video stream

At high level, here are the steps followed.

 - Train the model
	 - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier
	 - Optionally, apply color transforms (binned color features, histograms of color) and append them to your HOG feature vector. 
	 - Train a classifier on normalized extracted features
 - Use trained model to search / detect vehicle
	 - Implement a sliding-window technique and use trained classifier to search for vehicles in images.
	 - Run the pipeline on a video stream, create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
	 - Estimate a bounding box for vehicles detected.

The project has been implemented in traditional CV as well as deep learning methods.

This project is step 2 of the overall implementation

1. First, the advanced lane-finding (Project 4)
2. Second, the vehicle detection and tracking (Project 5)

With this, as I understand, Udacity is trying to subtly introduce students to syllabus in term 2 (simultaneous localisation and mapping – SLAM) using camera sensors.

----------
 **2. How to run**
-------------
**Computer Vision Method**

Here are the list of files used

 - cvcar_detect. ipynb: Defines steps to extract required features from an image, use classifier to detect cars and to generate a video
 - cvconfig.py : Global static parameters used across files
 - cvfeatures.py 	: Functions defined to extract features to train the
   model 
 - cvfind_cars.py 	: Functions defined to find cars in an image,
   given the trained model 
 - cvtrain.py : Functionality to train the SVM classifier

Steps to execute:

1. Run ‘cvtrain.py’ to train the SVM classifier and to save trained weights.
2. Follow instructions as in ‘cvcar_detect.ipynb’ to extract features, to identify cars in an image and finally to process video.

**Deep Learning Method**

Here are the list of files used

 - dncar_detect. ipynb: Defines steps to extract required features from an image, use classifier to detect cars and to generate a video
 - dnmodel_def.py : Defines model and further pre-trained weights loaded here
 - dntrain.py : Functionality to train the convolutional network

Steps to execute:

1. Run ‘dntrain.py’ to train the model and to save trained weights.
2. Follow instructions as in ‘dncar_detect.ipynb’ to extract features, to identify cars in an image and finally to process video.

----------
**3. Understanding Data**
-------------
Udacity has provided 8,792 images of car and 8,968 images of non-cars. The images have 64 x 64 pixels. 
From this, we can verify that dataset is well balanced, i.e., have as many positive as negative examples, or in the case of multi-class problems, roughly the same number of cases of each class.
This is required to avoid having algorithm simply classify everything as belonging to the majority class.

Here are two samples from the training dataset for car / not-car images.

![car_notcar](https://cloud.githubusercontent.com/assets/17127066/23933680/2fd00f7a-0966-11e7-8173-3f20d8314f6f.png)

----------
 **4. Train the model**
-------------
**Extract features**

To detect a vehicle in an image, we need to understand the features that differentiates the vehicle from rest of the image. Features capturing color and shape information should help in detection. Combination of below features were considered for this project.

Pixel intensity (spatial binning to capture shape and color) 

- Useful to detect things that do not vary in their appearance much. However, not so useful if objects appear in different forms, orientation or sizes.
- Binning size considered is 32
- length of feature vector: 32 * 32 * 3 = 3072

Histogram of pixel intensity (captures color)

- This transform is used to compute color values in an image.
- Histogram of a known image is compared with regions of the test image. Locations with similar color distribution will result as a close match
- Here, dependence on the structure is removed and match solely depends on the distribution of color values. Hence, the search might result in a lot of false positives.
- length of feature vector: 32 + 32 + 32 = 96

![histogram1](https://cloud.githubusercontent.com/assets/17127066/23933684/3038d762-0966-11e7-9c02-559e2fa46d52.png)
![histogram2](https://cloud.githubusercontent.com/assets/17127066/23933685/30404ef2-0966-11e7-8655-0d35dc612f42.png)

Principal Component Axis (PCA)

- Number of components considered: 3
- Length of the feature vector: 196
- With PCA, it takes longer time to train the classifier and to process the video. Also, there is no significant improvement with test accuracy. Hence, this feature was not included to train the classifier.

Histogram of Oriented Gradients (HOG) / Gradients of pixel intensity (captures shape for structural cues)

 - Since cars are a class of objects that vary in color, we would need structural ques as a feature
 - To extract the optimal HOG features, experimented with different color spaces and tweaked different parameters. Via trial-and-error, iterated through multiple loops of HOG feature parameter tweaking, visualizing static images, visualizing the final effect on video, more HOG feature parameter tweaking, etc.
 - After much experimentation, settled on the following HOG feature extraction parameters:
	 - Color space: ‘YCrCb’ 
	 - Channel: (all) 
	 - Orientations: 9 
	 - Pixels per cell: 8 
	 - Cells per block: 2
 - Length of feature vector: 5292
 - Code to extract HOG features is in the function get_hog_features() in the file 'cvfeatures.py'. It uses scikit-learn's hog() function.
 - The above parameters are hard-coded in the file 'cvconfig.py'.

![hog1](https://cloud.githubusercontent.com/assets/17127066/23933686/30465d92-0966-11e7-9109-8e0fd0c038a9.png)
![hog2](https://cloud.githubusercontent.com/assets/17127066/23933687/304c7ce0-0966-11e7-834c-d44b797f7fe4.png)

**Learning Algorithm**

![classifier2](https://cloud.githubusercontent.com/assets/17127066/24052648/3d5c99c2-0b5c-11e7-8b91-211b2bd12d73.png)

A classic approach is to first design a classifier that can differentiate car images from non-car images, and then run that classifier across an entire frame sampling small patches along the way.
The patches that classified as car are the desired detections.
For this approach to work properly, trained classifier must be able to distinguish car and non-car images.

Pre-processing steps

 - Random Shuffling of the data : To avoid problems due to ordering of
   the data 
 - Splitting the data into a training and testing set : To avoid
   overfitting / improve generalization 
 - Normalization of features, using Scaler to zero mean and unit variance : To avoid individual features or sets of features dominating the response of the classifier. Average and stdev are calculated independently for each individual feature in the feature vector.

Choice of classifier

 - SVC with RBF kernel - Takes longer time to train and predict. Also
   takes longer to process viseo stream 
 - SGD Classifier - Works similar to LinearSVC, but found that this classifier responds with more false positive results 
 - KerasClassifier - This is a Keras wrapper that implements Scikit-Learn classifier interface. Works quite well, however could not find a consistent way to save weights. 
 - LinearSVC – This classifier works well when compared to the above list, is implemeented for this project.

SVM Classifier

- LinearSVM used as classifier
- Color space used - 'YCrCb'
- Despite the high accuracy there is a systematic error as can be seen from investigating the false positive detections. The false positives include frequently occurring features, such as side rails, lane markers etc. 
- Feature vector length: 8460
- Test Accuracy of SVC =  0.9916

----------
 **5. Search / detect vehicle**
-------------
A variant of template matching is implemented through sliding windows. Image is stepped through in a grid pattern and looked to extract the same features that trained the classifier on each window. Window is slided through the region of interest across the image to search for vehicles using a trained classifier. Classifier is run to give a prediction at each step. Classifier responds if any of the windows in the image contains cars.

 - Sliding Window Search Vs Hog Sub-sampling 
	 - With ‘sliding window search’, HOG features (and other features) are extracted from each individual window as images are searched for cars. This is very inefficient in terms of performance/time.  Experimented with 32, 48, 64, 96, 128 window sizes, found 48 and 64 sizes returns with good results. However, is inefficient while running  the video streams. 
	 - As an alternative, in sub-sampling method, features are extracted just once for the entire region of interest. Extracted array is then sub-sampled for each sliding window. With the trials for 48, 64 window sizes, found both return similar results while with 48-sized windows is a bit slow. Sub-sampling method works with good accuracy and is also very efficient. 
	 - Tried combining above methods, but found just running sub-sampling method works well while processing video streams. 

 - Increase video processing speed - Since there is a lot of correlation between frames, tried adopting windows search once in 10 frames. Found this not working so well and instead measuring every frame, keeping it lean is preferred.
 - Taking average over last 10 frames to get smooth frames - This option worked well and is retained in the implementation
 - Pickling hot windows detection (after running SVM for all frames) so that heatmap algorithm and car detection runs in fast loop - This would not generalize well while processing other video streams, hence not implemented.

Below is an example of running sliding window search on an image (blue boxes indicate a vehicle detection in that window). As we can see, there are many boxes detected as vehicles, even though not all boxes are vehicles. Density of boxes tend to be high around actual vehicles, so we can take advantage of this fact when predicting the final bounding box.

![sub_sampled](https://cloud.githubusercontent.com/assets/17127066/23933688/30682d82-0966-11e7-92b3-cf51759f7cc4.png)

Heat map is created by adding the contributions of each predicted bounding box, similar to the method presented in the lectures.

![heatmap](https://cloud.githubusercontent.com/assets/17127066/23933682/302c7896-0966-11e7-8302-6242bd32742c.png)

Further, heatmap is thresholded to a binary image.

![heatmap_thres](https://cloud.githubusercontent.com/assets/17127066/23933683/30331ce6-0966-11e7-8378-4e2523bd9436.png)

scikit-learn's label() function is used to draw the final bounding boxes based on thresholded heatmap.

![final_tight_bound](https://cloud.githubusercontent.com/assets/17127066/23933681/300ca282-0966-11e7-8de9-ea39cc92cb1d.png)


----------
**6. Video Implementation**
-------------
Developed pipeline is applied to generate a video. Output 'cv_out.mp4' can be found in the current github repository.

It  took a lot of time to get a decent video out and some measures taken to reduce false positives are:

1. Playing around with heatmap threshold as explained in lectures
2. Augmenting data helped quite a bit. Tried out 'Flipping an image horizontally', 'Translating an image' and 'Enhancing brightness' in multiple combinations and finally retained image flipping.
3. Successive frames are integrated for smooth transition.

----------
**7. Discussion**
-------------
The main challenge for this project was parameter tuning, mostly to reduce the number of false positives in the video. Even though the HOG+SVM classifier reported good test accuracy after training, it did not necessarily mean good results in the overall vehicle detection task.

The project could be enhanced to also measure the distance between cars.

Vehicles are identified on the other side of the road and could be avoided.


----------
**8. Solved using Deep Learning**
-------------

Have also attempted to solve this problem using Deep Learning, replacing HOG+SVM pipeline. There are several advantages with this approach such as better performance with usage of GPU
and avoid playing with all the parameters related to feature extraction

Given the system limitation (pentium i5, 4GM RAM  and 2GB GPU), had to define a manageable model. Explored YOLO, SSD, U-Net and finally arrived at the custom model using pre-trained YOLO_TINY weights.

- Lower initial convolutional layers are loaded with yolo_tiny pre-trained weights and trainable is set to False. Higher layers are trained with Udacity data. This helped to reduce weight size from 180 MB to manageable 25MB
- Have retained overall structure that was used in traditional CV model. Just  the HOG+SVM functionality has been replaced deep learning
- A few of False Positives are retained to get a sense of what  deep learning identifies and understand the meaning behind.
- The network does not recognize the entire car and usually bounded box can be found on the rear part. The reason could be that most of images in the training set contains only the rear portion of the car.
- Sometime, network identifies traffic signs with numbers. In this case, network could be assuming that to be number plate of the car.

Referring to the example below, this image is passed through the pipeline.
![dnimage](https://cloud.githubusercontent.com/assets/17127066/24051960/d603b726-0b59-11e7-9d9d-e261a4d90dd6.png)

Here, cars are identified and marked with mesh-grids.
![dnmeshed](https://cloud.githubusercontent.com/assets/17127066/24051961/d608844a-0b59-11e7-9311-d351819fcb74.png)

Heatmap is created using similar function that was used in CV method
![dnheatmap](https://cloud.githubusercontent.com/assets/17127066/24051958/d5f0f62c-0b59-11e7-9fce-e9607ac8516c.png)

Thresholded heatmap

![dnheatthresh](https://cloud.githubusercontent.com/assets/17127066/24051959/d5f9c84c-0b59-11e7-8c73-144dc77bbf52.png)

Finally, original image augmented with bounding boxes based on thresholded heatmap.
![dntightbound](https://cloud.githubusercontent.com/assets/17127066/24051962/d6104cfc-0b59-11e7-84cd-b4f4bf1d1cf3.png)


----------
**9. Video implementation using Deep Learning**
-------------
Developed pipeline is applied to generate a video. Output 'dn_out.mp4' can be found in the current github repository.


----------
**10. References**
-------------

1. https://medium.com/@tuennermann/convolutional-neural-networks-to-find-cars-43cbc4fb713#.3kn6vl8om
2. https://medium.com/@xslittlegrass/almost-real-time-vehicle-detection-using-yolo-da0f016b43de#.21mqeepni
3. https://pjreddie.com/darknet/yolo/
