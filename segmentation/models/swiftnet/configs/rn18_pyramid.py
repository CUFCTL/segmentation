import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim
from pathlib import Path
import os
import numpy as np

from data.transform import *
from data.cityscapes import Cityscapes
from evaluation import StorePreds
from evaluation import ApplyColorMap
from models.semseg import SemsegModel
from models.resnet.resnet_pyramid import *
from models.loss import BoundaryAwareFocalLoss


from models.util import get_n_params

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path) # Path to save segmented images during evaluation (currently set to configs/out)
root = Path('datasets/cityscapes') #Path.home() / Path('datasets/Cityscapes')

evaluating = True
live_video = False
random_crop_size = 768

scale = 1
# The default mean and std are from the cityscapes dataset before nomralization (0-1)
# Will need to compute these values for other datasets like rellis (probably the entire dataset, not just train or test)
# DOES NOT NORMALIZE IMAGE BWTWEEN  0-1, ONLY PERFORMS STANDARDIZATION IN resnet_pyramid.py/forward()
mean = [73.15, 82.90, 72.3] # 73.15 / 255 = 0.2869, 82.60 / 255 = 0.3239 ... -> [0.2869, 0.3239, 0.2835]
std = [47.67, 48.49, 47.73]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))
# I am not sure why they are not normalizing from 0-1 and subtracting the mean and dividing by the std from the ImageNet dataset

num_classes = Cityscapes.num_classes
ignore_id = Cityscapes.ignore_id
class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

num_levels = 3
ostride = 4
target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // ostride, random_crop_size // ostride)

eval_each = 4
dist_trans_bins = (16, 64, 128)
dist_trans_alphas = (8., 4., 2., 1.)

if live_video:
    target_size = (640, 480) # (W, H)
    target_size_feats = (640 // ostride, 480 // ostride)
else:
    target_size = (2048, 1024)
    target_size_feats = (2048 // ostride, 1024 // ostride)

trans_val = Compose(
    [Open(),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Tensor(),
     ]
)

if evaluating:
    trans_train = trans_train_val = trans_val
else:
    trans_train = Compose(
        [Open(),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=ignore_id, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         LabelDistanceTransform(num_classes=num_classes, reduce=True, bins=dist_trans_bins, alphas=dist_trans_alphas),
         Tensor(),
         ])

dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val')
#dataset_test = Cityscapes(root, transforms=trans_val, subset='test')

interpolation_type = 'bilinear' #'bicubic' # This has been changed to binlinear because bicubic is not supported by TensorRT

backbone = resnet18(pretrained=True,
                    pyramid_levels=num_levels,
                    k_upsample=3,
                    scale=scale,
                    mean=mean,
                    std=std,
                    k_bneck=1,
                    pyramid_subsample=interpolation_type,
                    output_stride=ostride,
                    efficient=False) ############ efficient should = True when not converting to onnx - need to test with this set to False
model = SemsegModel(backbone, num_classes, k=1, bias=True)
if evaluating:
    model_path = 'weights/rn18_pyramid/model_best_one_input.pt'
    print(f"\nEvaluating using {model_path.split('/')[-1]}\n")
    model.load_state_dict(torch.load(model_path), strict=False)
else:
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)

bn_count = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
print(f'Num BN layers: {bn_count}')

if not evaluating:
    lr = 4e-4
    lr_min = 1e-6
    fine_tune_factor = 4
    weight_decay = 1e-4
    epochs = 250

    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)


if evaluating:
    batch_size = bs = 1 
    print(f'Batch size: {1}')
else:
    batch_size = bs = 14
    print(f'Batch size: {bs}')
nw = 4

#print(dataset_train[0]['image'].shape)
#print(dataset_train[0]['labels'].shape)

loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=nw)
if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate, num_workers=nw)
    #loader_train = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=nw)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=nw, pin_memory=True,
                              drop_last=True, collate_fn=custom_collate, shuffle=True)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

if evaluating and not live_video:
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    store_dir = f'{dir_path}/out/'
    for d in ['', 'val', 'train']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(color_info)
    to_image = Compose([Numpy(), to_color])
    eval_observers = [StorePreds(store_dir, to_image, to_color)]
elif evaluating and live_video:
    eval_loaders = [(loader_val, 'val')]
    to_color = ColorizeLabels(color_info)
    to_image = Compose([Numpy(), to_color])
    eval_observers = [ApplyColorMap(to_image, to_color)]
