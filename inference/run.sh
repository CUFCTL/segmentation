#!/usr/bin/env bash

python inference.py  -r\
    --mode video \
    --onnx_file models/model_cityscapes_h480_w640.onnx \
    --video_path media/riggs_h480_w640.mp4 \
    --cmap cityscapes