[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# The Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

# Overview of Files
My project includes the following files:
* [README.md](https://github.com/MarkBroerkens/CarND-Advanced-Lane-Lines/blob/master/README.md) (writeup report) documentation of the results 

[//]: # (Image References)
[image1]: ./output_images/vehicle_non_vehicle.png
[image2]: ./output_images/vehicle_hog1_orient8_pix_per_cell8_cell_per_block2_YCrCb.jpg
[image2a]: ./output_images/non_vehicle_hog6_orient8_pix_per_cell8_cell_per_block2_YCrCb.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4



# Histogram of Oriented Gradients (HOG)

## Feature extraction from the training images

The code for this step is contained in the file called `hog.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image2a]

## Histogram parameters
The combinations of parameters and its impact on the test accuracy is claculated in test `test_01_train_histogram()` of `test_train.py` 

| Number of features | Color space | Numer of bins | Test Accuracy |
| -----------------------|---------------|------------------|------------------|
|24 | RGB | 8 | 0.842 |
|48 | RGB | 16 | 0.884 |
|96 | RGB | 32 | 0.902 |
|192 | RGB | 64 | 0.922 |
|24 | HSV | 8 | 0.906 |
|48 | HSV | 16 | 0.946 |
|96 | HSV | 32 | 0.947 |
|192 | HSV | 64 | **0.952** |
|24 | LUV | 8 | 0.868 |
|48 | LUV | 16 | 0.889 |
|96 | LUV | 32 | 0.910 |
|192 | LUV | 64 | 0.932 |
|24 | HLS | 8 | 0.888 |
|48 | HLS | 16 | 0.921 |
|96 | HLS | 32 | 0.948 |
|192 | HLS | 64 | **0.961** |
|24 | YUV | 8 | 0.865 |
|48 | YUV | 16 | 0.889 |
|96 | YUV | 32 | 0.911 |
|192 | YUV | 64 |  0.930 |
|24 | YCrCb | 8 | 0.873 |
|48 | YCrCb | 16 | 0.883 |
|96 | YCrCb | 32 | 0.923 |
|192 | YCrCb | 64 | 0.936 |

The following parameters resulted in the best test accuracy: color space HLS or HSV with 64 bins which resulted in a test accuracy of more than 95%.

The test was executed twice in order to validate the results.


## Spatial parameters
| Number of features | Color space | Spartial Size | Test Accuracy |
| -----------------------|---------------|------------------|------------------|
|192 | RGB | 8 | 0.902 |
|768 | RGB | 16 | 0.905 |
|3072 | RGB | 32 | 0.909 |
|12288 | RGB | 64 | 0.905 |
|192 | HSV | 8 | 0.873 |
|768 | HSV | 16 | 0.899 |
|3072 | HSV | 32 | 0.872 |
|12288 | HSV | 64 | 0.877 |
|192 | LUV | 8 | 0.898 |
|768 | LUV | 16 | 0.926 |
|3072 | LUV | 32 | 0.901 |
|12288 | LUV | 64 | 0.904 |
|192 | HLS | 8 | 0.871 |
|768 | HLS | 16 | 0.895 |
|3072 | HLS | 32 | 0.870 |
|12288 | HLS | 64 | 0.877 |
|192 | YUV | 8 | 0.901 |
|768 | YUV | 16 | **0.919** |
|3072 | YUV | 32 | 0.901 |
|12288 | YUV | 64 | 0.896 |
|192 | YCrCb | 8 | 0.899 |
|768 | YCrCb | 16 | **0.924** |
|3072 | YCrCb | 32 | 0.896 |
|12288 | YCrCb | 64 | 0.901 |

The following parameters resulted in the best test accuracy (>92%): color space YCrCb with spatial size of 16.
The test was executed twice in order to validate the results.


## HOG parameters.

| len features | color_space | orient | pix_per_cell | cell_per_block | hog_channel | accuracy | 
| 2940 | HSV | 5 | 8 | 2 | ALL | 0.962 |
| 6000 | HSV | 5 | 8 | 4 | ALL | 0.965 |
| 540 | HSV | 5 | 16 | 2 | ALL | 0.971 |
| 240 | HSV | 5 | 16 | 4 | ALL | 0.969 |
| 5292 | HSV | 9 | 8 | 2 | ALL | 0.955 |
| 10800 | HSV | 9 | 8 | 4 | ALL | 0.950 |
| 972 | HSV | 9 | 16 | 2 | ALL | 0.973 |
| 432 | HSV | 9 | 16 | 4 | ALL | 0.969 |
| 7644 | HSV | 13 | 8 | 2 | ALL | 0.942 |
| 15600 | HSV | 13 | 8 | 4 | ALL | 0.945 |
| 1404 | HSV | 13 | 16 | 2 | ALL | 0.968 |
| 624 | HSV | 13 | 16 | 4 | ALL | 0.968 |
| 2940 | LUV | 5 | 8 | 2 | ALL | 0.967 |
| 6000 | LUV | 5 | 8 | 4 | ALL | 0.968 |
| 540 | LUV | 5 | 16 | 2 | ALL | **0.976** |
| 240 | LUV | 5 | 16 | 4 | ALL | **0.975** |
| 5292 | LUV | 9 | 8 | 2 | ALL | 0.965 |
| 10800 | LUV | 9 | 8 | 4 | ALL | 0.966 |
| 972 | LUV | 9 | 16 | 2 | ALL | **0.976** |
| 432 | LUV | 9 | 16 | 4 | ALL | **0.977** |
| 7644 | LUV | 13 | 8 | 2 | ALL | 0.969 |
| 15600 | LUV | 13 | 8 | 4 | ALL | 0.969 |
| 1404 | LUV | 13 | 16 | 2 | ALL | **0.983** |
| 624 | LUV | 13 | 16 | 4 | ALL | **0.976** |
| 2940 | HLS | 5 | 8 | 2 | ALL | 0.957 |
| 6000 | HLS | 5 | 8 | 4 | ALL | 0.966 |
| 540 | HLS | 5 | 16 | 2 | ALL | **0.972** |
| 240 | HLS | 5 | 16 | 4 | ALL | 0.962 |
| 5292 | HLS | 9 | 8 | 2 | ALL | 0.949 |
| 10800 | HLS | 9 | 8 | 4 | ALL | 0.952 |
| 972 | HLS | 9 | 16 | 2 | ALL | **0.972** |
| 432 | HLS | 9 | 16 | 4 | ALL | 0.968 |
| 7644 | HLS | 13 | 8 | 2 | ALL | 0.944 |
| 15600 | HLS | 13 | 8 | 4 | ALL | 0.949 |
| 1404 | HLS | 13 | 16 | 2 | ALL | 0.967 |
| 624 | HLS | 13 | 16 | 4 | ALL | 0.969 |
| 2940 | YUV | 5 | 8 | 2 | ALL | 0.965 |
| 6000 | YUV | 5 | 8 | 4 | ALL | 0.969 |
| 540 | YUV | 5 | 16 | 2 | ALL | **0.976** |
| 240 | YUV | 5 | 16 | 4 | ALL | **0.978** |
| 5292 | YUV | 9 | 8 | 2 | ALL | 0.967 |
| 10800 | YUV | 9 | 8 | 4 | ALL | **0.975** |
| 972 | YUV | 9 | 16 | 2 | ALL | **0.980** |
| 432 | YUV | 9 | 16 | 4 | ALL | **0.978** |
| 7644 | YUV | 13 | 8 | 2 | ALL | **0.980** |
| 15600 | YUV | 13 | 8 | 4 | ALL | **0.970** |
| 1404 | YUV | 13 | 16 | 2 | ALL | **0.981** |
| 624 | YUV | 13 | 16 | 4 | ALL | **0.973** |
| 2940 | YCrCb | 5 | 8 | 2 | ALL | 0.964 |
| 6000 | YCrCb | 5 | 8 | 4 | ALL | 0.969 |
| 540 | YCrCb | 5 | 16 | 2 | ALL | **0.976** |
| 240 | YCrCb | 5 | 16 | 4 | ALL | **0.980** |
| 5292 | YCrCb | 9 | 8 | 2 | ALL | 0.967 |
| 10800 | YCrCb | 9 | 8 | 4 | ALL | 0.967 |
| 972 | YCrCb | 9 | 16 | 2 | ALL | **0.976** |
| 432 | YCrCb | 9 | 16 | 4 | ALL | **0.976** |
| 7644 | YCrCb | 13 | 8 | 2 | ALL | **0.970** |
| 15600 | YCrCb | 13 | 8 | 4 | ALL | **0.974** |
| 1404 | YCrCb | 13 | 16 | 2 | ALL | **0.977**|
| 624 | YCrCb | 13 | 16 | 4 | ALL | **0.979**|

The following parameters resulted in the best test accuracy (>98%) and reasonable effort: color space YUV, number of orientations=5, pix per cell 16, cells per block 2, and hog channels = "All".

## Parameters
color space YCrCb, number of orientations=5, pix per cell 16, cells per block 4, and hog channels = "All", 
spartial size of 16, 64 bins.

| len features | spatial_feat | hist_feat | hog_feat | accuracy |
|---------------|--------------|------------|-----------|------------|
| 1056 | True | True | True | 0.986 |
| 816 | True | True | False | 0.950 |
| 1008 | True | False | True | 0.975 |
| 768 | True | False | False | 0.926 |
| 288 | False | True | True | 0.985 |
| 48 | False | True | False | 0.884 |
| 240 | False | False | True | 0.980 |

The following parameters resulted in the best test accuracy (98,6%): 
no spartial, enabled hist and hog features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

[Test data of vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
[Test data of non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

https://github.com/udacity/self-driving-car/tree/master/annotations


# Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

# Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

# Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

