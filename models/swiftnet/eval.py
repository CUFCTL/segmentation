"""eval.py
Main inference script.

To run inference on a validation/test set:
    - In configuration file set evaluting=True 
    - In configuration file set live_video=False 
    - In configuration file set the target_size and target_size_feats to the shape of the image 
      (usually the full size of the image)
    - In configuration file set model_path to the file path of trained model to inference on
    - In eval.sh uncomment the line with 'static' and comment the line wtih 'live'
    

To run inference on a live video feed:
    - In configs/rn18_pyramid.py set evaluting=True 
    - In configs/rn18_pyramid.py set live_video=True 
    - In configs/rn18_pyramid.py set the target_size and target_size_feats to the shape of the image 
      (usually the full size of the image)
    - In eval.sh uncomment the line with 'live' and comment the line wtih 'static'
        - If you would like to time the fps uncomment the line with '--timing'

Usage: bash eval.sh
"""
import argparse
from pathlib import Path
import importlib.util
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.cityscapes import CityscapesVideo
from data.transform import *
from evaluation import evaluate_semseg
from evaluation import evaluate_semseg_timing
from evaluation import evaluate_semseg_live_video


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('mode', type=str, default='static', help='Inference a set of images or live video feed (mode=live)')
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')
parser.add_argument('--timing', dest='timing', action='store_true', help='Time the inference speed and do not save images')

if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    current_device = torch.cuda.current_device()
    print(f'Current cuda device: {current_device}')
    print(f'Current cuda device name: {torch.cuda.get_device_name(current_device)}')
    model = conf.model.cuda()

    mode = args.mode
    print(mode)
    if mode == 'live':

        vid = cv2.VideoCapture(0)
        count = 0

        target_size = conf.target_size
        target_size_feats = conf.target_size_feats

        trans_video = Compose(
                [OpenVideo(),
                SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
                Tensor(),
                ])

        while(True):

            # Capture the video frame
            _, frame = vid.read()

            # Wait for key press for 10ms and return the keys value
            k = cv2.waitKey(10)
            
            dataset_video = CityscapesVideo(frame, transforms=trans_video, subset='video')
            loader_video = DataLoader(dataset_video, batch_size=1, collate_fn=custom_collate, num_workers=4)
            class_info = dataset_video.class_info
            #print("Here")
            if k == ord('q'): # 'q'
                break

            segmented_frame = evaluate_semseg_live_video(model, loader_video, class_info, observers=conf.eval_observers, eval_per_steps=1)
            
            combined_image = np.concatenate((frame, segmented_frame[0]), axis=1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fps_text = f'FPS:'# {fps}'
            cv2.putText(combined_image, fps_text, (5, 30), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', combined_image)

        vid.release()
        cv2.destroyAllWindows()

    elif mode == 'static':
        class_info = conf.dataset_val.class_info
        for loader, name in conf.eval_loaders:
            if args.timing:
                evaluate_semseg_timing(model, loader, class_info, observers=conf.eval_observers, eval_per_steps=1)
            else: # Default 
                iou, per_class_iou = evaluate_semseg(model, loader, class_info, observers=conf.eval_observers)
                print(f'{name}: {iou:.2f}')
