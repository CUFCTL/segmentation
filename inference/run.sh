#!/usr/bin/env bash

python inference.py \
    --mode video \
    --onnx_file models/model_cityscapes_h480_w640.onnx \
    --video_path media/go-pro-1.MP4 \
    --cmap cityscapes