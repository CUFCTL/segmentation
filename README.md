# Semantic Segmentation for the VIPR Project
This is our semantic segmentation pipeline for the VIPR project. The repository is capable of training/evaluating semantic segmentation models, generating ONNX files, and performing live inference on a camera feed. Inference supports live video feed, a recorded `.mp4` video, and a single image. Due to the many dependencies, and the requirements of ROS, setup instructions are only included for Docker (GPU enabled).

### Inference Preview
https://user-images.githubusercontent.com/34605638/203414948-aea30ddd-0e74-461a-bdc0-b607b3e82f7b.mp4

### Table of Contents
* [Launching the Docker Container](#launching-the-docker-container)

## Launching the Docker Container
The GPU enabled docker container has all the dependencies for this project so the user only needs [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (nvidia-docker2) installed. NVIDIA does not currently support GPU use in docker containers on Windows so these instructions are __only for Linux__. NVIDIA does offer very good suport for GPU Docker usage with WSL 2, however, this project has not been tested on it. Lastly, the GPU must support [compute capability](https://developer.nvidia.com/cuda-gpus) 6.0 or higher due to PyTorch constraints.
```bash
docker pull bselee/vipr
```

# **  Documentation still in progress **
