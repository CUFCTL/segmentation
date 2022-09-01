#!/bin/sh

python train.py configs/rn18_pyramid.py --store_dir=weights
#CUDA_LAUNCH_BLOCKING=1 python train.py configs/rn18_pyramid.py --log --store_dir=weights