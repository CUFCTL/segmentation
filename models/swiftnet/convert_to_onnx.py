"""Convert trained model to ONNX format

Set up:
    1) In configs/rn18_pyramid.py set evaluating = True 
    2) In configs/rn18_pyramid.py set the efficient parameter in resnet18() to false 
            - onnx does not support this type of checkpointing
    2) Set cubic interpolation to bilinear interpolation
            - TensorRT and OpenCV does not support cubic interpolation
    3) In configs/rn18_pyramid.py set model_path variable to the .pt weight file to convert

Usage 1: python convert_to_onnx.py configs/rn18_pyramid.py model.onnx
Usage 2: python convert_to_onnx.py configs/rn18_pyramid.py model.onnx --height 480 --width 640
            
"""

import os
import argparse
import importlib.util
import torch.onnx
from data import Cityscapes
from pathlib import Path

parser = argparse.ArgumentParser(description='ONNX converter')
# -- makes the argument optional
parser.add_argument('config', type=str, default='configs/rn18_pyramid.py', help='Path to configuration .py file')
parser.add_argument('onnx_name', type=str, help='Name of onnx model ending in .onnx')
parser.add_argument('--height', type=int, default=1024, help='Height of the evaluation input image')
parser.add_argument('--width', type=int, default=2048, help='Width of the evaluation input image')


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_to_onnx(model, store_dir, batch_size, height, width):
    model.eval()

    dummy_input = torch.randn(batch_size, 3, height, width)
    output_stride = 4

    #dataset_val = Cityscapes(root, transforms=trans_val, subset='val')

    #logits, additional = model.do_forward(dummy_input, dummy_input.shape[1:3])
    image_size = dummy_input.shape[2:4]
    target_size = (image_size[0] // output_stride, image_size[1] // output_stride) 
    #print(image_size)
    #print(target_size)
    logits, additional = model(dummy_input)#, image_size)

    print(f'\nONNX model saved in: {store_dir}')

    # Export the model   
    torch.onnx.export(model,         # model being run 
         (dummy_input),# image_size),       # model input (or a tuple for multiple inputs) 
         store_dir,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'\nCreating ONNX file with input image size {args.height}x{args.width} (HxW)...')

    print('Loading model')
    conf = import_module(args.config)

    class_info = conf.dataset_val.class_info
    
    model = conf.model # The model weights, .pt, is loaded inside the config file so we don't need to do it here
    print(f'Using batch size of {conf.batch_size}')

    file_name = args.onnx_name
    if (args.width != 2048 and args.height != 1024):
        path_name = Path(args.onnx_name).stem
        file_name = path_name + '_h' + str(args.height) + '_w' + str(args.width) + '.onnx'

    store_dir = os.path.join(os.path.dirname(conf.model_path), file_name)#Path(conf.model_path).stem + '.onnx')
    convert_to_onnx(model, store_dir, conf.batch_size, args.height, args.width)

