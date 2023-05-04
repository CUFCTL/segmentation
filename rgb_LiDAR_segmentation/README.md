# Semantic Segmentation and LiDAR Fusion for the VIPR Project
This is our semantic segmentation pipeline and LiDAR point cloud pipeline for the VIPR project. The repository is capable of training/evaluating semantic segmentation models, joining point clounds and images and class condensation. 


## Table of Contents
* [Overview](#overview)
* [Setup and Deployment](#setup-and-development)
* [Preparing Datsets](#preparing-datasets)
* [Training](#Training-the-model)

## Overview

### Installation
These scripts are runned in matlabR2022a and require computer vision and deep learning matlab app's and there dependecies. The base model used is a resnet18 which requires the resnet app to be install too. when running the train.m matlab command window will show required apps.

###

### Features
This projects features is to train/evaluate a semantic segmentation model for the rellis-3d dataset.  This is able to condense classes of the set and utalize both its rgb image data and LiDAR point cloud data for the segmentation.

![Alt text](/images/point_cloud.jpg)

### Training the model
To train the model you need to run the train.m file and provide a path to the saved rellis-3d dataset. This will run the normal image segmentation without LiDAR.

To train the model on condensed classes you need to un-comment 
classes = condensed_classes;
labelIDs = condensed_labelIDs;
cmap = condensed_cmap;
at the bottom of the first section where the classes get defined.

To train the model on fused LiDAR and RGB data you first need to create a train and test set of the depth images. Run the lidar_rellis.m file and provide the correct paths to the point clouds and the images.  This will create a folder of depth images for the training and testing. Next the NN model will require some modification to be able to run these new images.  Load in the Rellis model into the workspace then open the Deep Network Designer tool.  Next load the model in this tool and you will need to create two new layers to replace the first two.  First create a new imageInputLayer and make the input size 1200,1920,4 to accept the new depth layer.  Then create a conv2dLayer and replace the top two default layers with these new ones. After this change the image size varable in the train.m from [1200 1920 3] to [1200 1920 4]. The rest of the train and test will run normally.


### Testing the model
The second half of the train.m script contains the testing functions.  First it will show you a image from the test set which has been segmentated and overlayed with a colormap to show the infrence.  It will then show the auctal against the expected segmentation as a comparison with the GT. Next it will calculate the jaccard index which is the miou score and display the mean iou and the iou for each indivigual class.  Next it runs over the test set again and generates all of the class metrics. Then it performs a timing test, the model will loop through and infrence on the test set and be timed to get a average infrence time of the model.  The remaining segments allow for testing of single images or generating graphs like confusion matrices.

