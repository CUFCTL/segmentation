# Semantic Segmentation for the VIPR Project
This is our semantic segmentation pipeline for the VIPR project. The repository is capable of training/evaluating semantic segmentation models, generating ONNX files, and performing live inference on a camera feed. Inference supports live video feed, a recorded `.mp4` video, and a single image. Due to the many dependencies, and the requirements of ROS, setup instructions are only included for Docker (GPU enabled).

### Inference Preview
https://user-images.githubusercontent.com/34605638/203414948-aea30ddd-0e74-461a-bdc0-b607b3e82f7b.mp4

### Table of Contents
* [Launching the Docker Container](#launching-the-docker-container)
* [Deploying the Docker Container on the Husky](#deploying-the-docker-container-on-the-husky)

## Launching the Docker Container
The GPU enabled docker container has all the dependencies for this project so the user only needs [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (nvidia-docker2) installed. NVIDIA does not currently support GPU use in docker containers on Windows so these instructions are __only for Linux__. NVIDIA does offer very good suport for GPU Docker usage with WSL 2, however, this project has not been tested on it. Lastly, the GPU must support [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0 or higher due to PyTorch constraints.
```bash
docker pull bselee/vipr-segmentation:1.0
```
# **  Documentation still in progress **

## Deploying the Docker Container on the Husky
The docker container for general usage is slightly different than the docker container for the Husky robot. The main difference is the Husky docker container contains an inference/inference-ros.py script which instantiates the ROS node, subscribes to the compressed image ROS topic, and converts the image to work with OpenCV. This file is not in this GitHub repository at the moment, __in the future we should add it__. This means that to modify the inference-ros.py Husky docker container we need to modify the container directly and cannot use the Dockerfile. Instructions to modify this will be explained after the deployment instructions.

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
