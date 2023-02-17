#!/usr/bin/env bash

python inference.py  -r\
    --mode video \
    --onnx_file models/windows_camera_h720_w1280.onnx \
    --video_path media/outside_riggs.mp4 \
    --cmap cityscapes