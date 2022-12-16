#!/usr/bin/env bash

# Notes:
#   1. ipc=host allows multiple proccesses to communicate through shared memory
#      Exampe: it lets DataLoader num_workers > 0 work
#   

sudo xhost +si:localuser:root

docker run --gpus all -it --rm --privileged \
           --ipc host \
           --network host \
           --env DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           -v trained-models:/app/inference/models \
           -v trained-models:/app/models/swiftnet/weights \
           -v data:/app/models/swiftnet/datasets \
           bselee/vipr-segmentation:1.0 bash