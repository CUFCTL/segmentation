#!/usr/bin/env bash

# ipc=host allows multiple proccesses to communicate through shared memory
# Ex. it lets DataLoader num_workers > 0 work   

docker run --gpus all -it --rm \
           --ipc=host \
           -v trained-models:/app/inference/models \
           -v data:/app/models/swiftnet/datasets \
           bselee/vipr-segmentation:1.0 bash