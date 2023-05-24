# Semantic Segmentation for the VIPR Project
This is our semantic segmentation pipeline for the VIPR project. The repository is capable of training/evaluating semantic segmentation models, generating ONNX files, and performing live inference on a camera feed. Inference supports live video feed, a recorded `.mp4` video, and a single image. Due to the many dependencies, and the requirements of ROS, setup instructions are only included for Docker (GPU enabled).

### Inference Preview
https://user-images.githubusercontent.com/34605638/203414948-aea30ddd-0e74-461a-bdc0-b607b3e82f7b.mp4

### [AuxAdapt](https://arxiv.org/abs/2110.12369) Method to Increase Temporal Consistency (Reduce Flickering)
https://user-images.githubusercontent.com/34605638/221033265-8567c556-dfa6-4b9b-be63-f7d73993a3ef.mp4

## Table of Contents
* [File Structure](#file-structure)
* [Local Setup and Development](#local-setup-and-development)
* [Launching the Docker Container](#launching-the-docker-container)
* [Features](#features)
* [Deploying the Docker Container on the Husky](#deploying-the-docker-container-on-the-husky)
* [Preparing Datsets](#preparing-datasets)
* [Publications](#publications)

## File Structure
	.
	├── inference                 # Live inference for .onnx files
	├── models                    # Train/test segmentation models (e.g., SwiftNet)
	├── rgb_LiDAR_segmentation    # DeepLabV3+ with RGB/LiDAR fusion (written by Max in MATLAB)
	├── scripts                   # Helper scripts disconnected from the pipeline
	├── testing                   # Archive of files which were never integrated into the pipeline (these should not be used)
	├── Dockerfile
	├── docker_run.sh	      # Script to run the local docker container
	└── environment.yml	      # File to setup the anaconda environment


## Local Setup and Development
This section covers setting up the project on a local machine for development and running the project in a docker container. If you just want to run the general usage docker container, to [Launching the Docker Container](#launching-the-docker-container).

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

## Launching the Docker Container
The GPU-enabled docker container has all the dependencies for this project so the user only needs [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (nvidia-docker2) installed. NVIDIA does not currently support GPU use in docker containers on Windows so these instructions are __only for Linux__. NVIDIA does offer very good suport for GPU Docker usage with WSL 2, however, this project has not been tested on it. Lastly, the GPU must support [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0 or higher due to PyTorch constraints.

First, clone the GitHub repository if you have not done so already.
```bash
git clone https://github.com/CUFCTL/segmentation.git
cd segmentation
```

Then, pull the general usage docker container.
```bash
docker pull bselee/vipr-segmentation:1.0
```

The docker container can now be started with
```bash
./docker_run.sh
```
When the docker conainter is exited, it is automatically deleted by the ```--rm``` flag in the ```docker_run.sh``` script.

### Modifying the Docker Container
The easiest way to modify the docker container is to first make the changes locally in the project repository, then build the Dockerfile.
```bash
docker build -t bselee/vipr-segmentation:1.0 .
```

After testing the new docker image, you can push it to Docker Hub with
```bash
docker push bselee/vipr-segmentation:1.0
```

## Features
The main uses of this project is to train/evaluate a semantic segmentation model and then perform inference on a live video stream.

After cloning the repository and setting up an Anaconda environment or docker container, you should have everything you need to run, modify, and create new features in this project. Remember to create a new branch everytime you modify or create a new feature.

### Training the model
Training a model is performed in the network specific directory, for example ```models/swiftnet/```

Modify a few variables in ```configs/rn18_pyramid.py```
 - set ```evaluting=False```
 - set ```live_video=False```
 - set the ```target_size``` and ```target_size_feats``` to the shape of the image 
   (usually the full size of the image)
 - set the ```root``` variable to the dataset root directory
 - set the desired batch size and number epochs (default epochs:250)

Begin training either by running the ```train.py``` file directly or running the ```train.sh``` bash script:
```bash
python train.py configs/rn18_pyramid_rellis.py --store_dir=weights
```
or
```bash
bash train.sh
```

### Testing the model
Training a model is performed in the network specific directory, for example ```models/swiftnet/```

Modify a few variables in ```configs/rn18_pyramid.py```
 - set ```evaluting=True```
 - set ```live_video=False```
 - set the ```target_size``` and ```target_size_feats``` to the shape of the image 
   (usually the full size of the image)
 - set ```model_path``` to the file path of trained model to inference on

Begin training either by running the ```train.py``` file directly or running the ```train.sh``` bash script:
```bash
python eval.py --timing configs/rn18_pyramid.py static
```
or
```bash
bash eval.sh
```


### Converting to ONNX format
Converting a model to ONNX format is performed in the network specific directory, for example ```models/swiftnet/```
```bash
cd models/swiftnet/
```

Modify a few variables in ```configs/rn18_pyramid.py```
 - set ```evaluating = True``` 
 - set ```model_path``` variable to the .pt weight file to convert

Convert the model to ONNX format by specifying the config file, output onnx file name, and the height/width of the desired inference images:
```bash
python convert_to_onnx.py configs/rn18_pyramid.py model.onnx --height 480 --width 640
```

If you wish to inference on different resolution images, a new ONNX file must be created.

### Inference on a live video feed
```bash
cd inference
./run.sh
```

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

## Preparing Datasets
TODO: write instructions on how to prepare Rellis-3D and Cityscapes before training

## Publications
[Semantic Segmentation with High Inference Speed in Off-Road Environments](https://www.sae.org/publications/technical-papers/content/2023-01-0868/)
<br>B Selee, M Faykus, M Smith
<br>SAE Technical Paper

[Utilizing Neural Networks for Semantic Segmentation on RGB/LiDAR Fused Data for Off-road Autonomous Military Vehicle Perception](https://www.sae.org/publications/technical-papers/content/2023-01-0740/)
<br>MH Faykus, B Selee, M Smith
<br>SAE Technical Paper

[Lossy Compression to Reduce Latency of Local Image Transfer for Autonomous Off-Road Perception Systems](https://ieeexplore.ieee.org/abstract/document/10020267)
<br>MH Faykus, B Selee, JC Calhoun, MC Smith
<br>2022 IEEE International Conference on Big Data (Big Data), 3146-3152

