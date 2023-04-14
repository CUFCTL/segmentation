# Semantic Segmentation for the VIPR Project
This is our semantic segmentation pipeline for the VIPR project. The repository is capable of training/evaluating semantic segmentation models, generating ONNX files, and performing live inference on a camera feed. Inference supports live video feed, a recorded `.mp4` video, and a single image. Due to the many dependencies, and the requirements of ROS, setup instructions are only included for Docker (GPU enabled).

### Inference Preview
https://user-images.githubusercontent.com/34605638/203414948-aea30ddd-0e74-461a-bdc0-b607b3e82f7b.mp4

### [AuxAdapt](https://arxiv.org/abs/2110.12369) Method to Increase Temporal Consistency (Reduce Flickering)
https://user-images.githubusercontent.com/34605638/221033265-8567c556-dfa6-4b9b-be63-f7d73993a3ef.mp4

## Table of Contents
* [Overview](#overview)
* [Local Setup and Development](#local-setup-and-development)
* [Deploying the Docker Container on the Husky](#deploying-the-docker-container-on-the-husky)
* [Preparing Datsets](#preparing-datasets)

## Overview
TODO

## Local Setup and Development
This section covers setting up the project on a local machine for development and running the project in a docker container. If you just want to run the general usage docker container, proceed to [Launching the Docker Container](#launching-the-docker-container).

### Installation
Clone the git repository
```bash
git clone https://github.com/CUFCTL/segmentation.git
```

### Anaconda environment
Create a virtual environment with the required dependencies and activate it.
```bash
conda env create -f environment.yml
source activate segmentation
```

### Features (Maybe delete this section or reword it)
After cloning the repository and setting up an Anaconda environment, you should have everything you need to run, modify, and create new features in this project. Remember to create a new branch everytime you modify or crete a new feature.

The main uses of this project is to train/evaluate a semantic segmentation model and then perform inference on a live video stream.

### Training the model
TODO

### Testing the model
TODO

### Converting to ONNX format
TODO

### Inference on a live video feed
TODO


### Launching the Docker Container (maybe put this section in a different order)
The GPU enabled docker container has all the dependencies for this project so the user only needs [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (nvidia-docker2) installed. NVIDIA does not currently support GPU use in docker containers on Windows so these instructions are __only for Linux__. NVIDIA does offer very good suport for GPU Docker usage with WSL 2, however, this project has not been tested on it. Lastly, the GPU must support [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0 or higher due to PyTorch constraints.
```bash
docker pull bselee/vipr-segmentation:1.0
```
# **  Documentation still in progress **

## Deploying the Docker Container on the Husky
The docker container for general usage is slightly different than the docker container for the Husky robot. The main difference is the Husky docker container contains an ```inference/inference-ros.py``` script which instantiates the ROS node, subscribes to the compressed image ROS topic, and converts the image to work with OpenCV. This file is not in this GitHub repository at the moment, __in the future we should add it__. This means that to modify the ```inference-ros.py``` Husky docker container we need to modify the container directly and cannot use the Dockerfile. Instructions to modify this will be explained after the deployment instructions.

To launch the docker container on the Husky, first pull the container on a laptop connected to the Husky's network:
```bash
docker pull bselee/vipr-segmentation:husky-deploy
```

Launch the docker container on the laptop, or whichever device you are using, 
```bash
sudo xhost +si:localuser:root # Enables docker GUI usage

sudo docker run --gpus all -it \
         --ipc host \
	 --network host \
         --env DISPLAY=$DISPLAY \
         bselee/vipr-segmentation:husky-deploy bash
```

Inside the docker container navigate to the inference directory and launch the segmentation with:
```bash
cd inference
python inference-ros.py \
  	--mode video \
	--onnx_file models/model_cityscapes_h480_w640.onnx \
	--cmap cityscapes

```
or use the ```run.sh``` script
```bash
./run.sh
```

The inference code is modular enough to load different onnx models and color maps. If you train another dataset, convert the new PyTorch model into an ONNX model, move the ONNX model to the ```models``` directory, and modify the ```--onnx_file``` command-line argument. . Similarly, if the segmentation color map is different, you can add teh color map to the ```inference/utils/utils.py``` file and change the ```--cmap``` argument.

### Modifying the Husky docker container
Like mentioned before, the Husky docker container cannot be modified just by changing this repository and rebuilding the image from the Dockerfile. You need to modify the container's code directly, commit and push the container to Docker Hub:

```bash
# Launch the docker container as shown above then modify code or add files

exit # exit the docker container
docker ps -a # find the id of the docker container we just exited
docker commit <container_id> bselee/vipr-segmentation:husky-deploy # commit the docker container
docker push bselee/vipr-segmentation # push the committed container to the Docker Hub Registry
```

The current registry is under my Docker Hub account (bselee) but __we should probably create a registry for the FCTL lab so anyone can push to it.__
# **  Documentation still in progress **

## Preparing Datasets
TODO

