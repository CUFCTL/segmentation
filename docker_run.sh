#!/usr/bin/env bash

docker run --gpus all -it --rm \
           -v trained-models:/app/inference/models \
           bselee/vipr-segmentation:1.0 bash