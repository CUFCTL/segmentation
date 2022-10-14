#!/usr/bin/env bash

# <model.onnx> => location of the onnx file to convert
# <model.trt> => location and name to save the new .trt file

# Usage: bash onnx_to_tensorrt.sh <model.onnx> <model.trt>

onnx_file="${1:-models/model_best.onnx}" # If command line arguments are not specified, give a default value
trt_file_name="${2:-models/model_best.trt}"

/usr/src/tensorrt/bin/trtexec --verbose --onnx=$onnx_file --saveEngine=$trt_file_name