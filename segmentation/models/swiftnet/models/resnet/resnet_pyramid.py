"""ResNet pyramid model for SwiftNet

Code taken from: https://github.com/orsic/swiftnet/blob/master/models/resnet/resnet_pyramid.py
Updated Paper: https://www.sciencedirect.com/science/article/pii/S0031320320304143
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from collections import defaultdict
from math import log2

from ..util import _UpsampleBlend

__all__ = ['ResNet', 'resnet18', 'resnet34', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def convkxk(in_planes, out_planes, stride=1, k=3):
    """kxk convolution with padding and no bias
    k // 2 gives optimal padding so the input is not downsampled when stride=1
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = norm(conv(x))
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    """utils.checkpoint is suppposed to make certain layers more efficient.
    Must have a layer that requires_grad=True
    """
    # return block(x)
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x) ############### I think i need to replace this function for onnx to work
    else:
        return block(x)


class Identity(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, input):
        return input


class BasicBlock(nn.Module):
    """BasicBlock for ResNet-18 & 34. This project only uses ResNet-18.
    SwiftNet uses a multibranch architecure which shares parameters of convolutional layers for each branch.
    For example, original_image and original_image/2 both get passed through the same self.conv1 layer, but 
    for batch normalization you do not want to share parameters so a list is created with pyramid size (number 
    of branches). This is expalined in section 4.1 of the paper.
    """
    
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, bn_class=nn.BatchNorm2d, levels=3):
        super().__init__()
        self.conv1 = convkxk(inplanes, planes, stride) # Downsamples here for layer 3_1, 4_1, and 5_1
        self.bn1 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.downsample = downsample  # used to downsample the residual
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels

    def forward(self, x, level):
        """level is passed from forward_resblock and is used to define the level of the image pyramid 
        (original or downsampled image). In the default case there are 3 levels: orginal, orignal/2 & orignal/4. 
        """
        residual = x

        # bn_x is a function that creates the layers and propagates the data 
        # bn_1 layers: conv -> BatchNorm2D -> relu 
        # bn_2 layers: conv -> BatchNorm2D  
        bn_1 = _bn_function_factory(self.conv1, self.bn1[level], self.relu_inp)
        bn_2 = _bn_function_factory(self.conv2, self.bn2[level])

        # out layers: same as bn_x layers but uses checkpointing which supposedly makes it more efficient
        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual 
        relu = self.relu(out)

        return relu, out
    
    # I think this isn't used
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
                                      unexpected_keys, error_msgs)
        missing_keys = []
        unexpected_keys = []
        for bn in self.bn1:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)
        for bn in self.bn2:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)
        
                
class ResNet(nn.Module):
    
    def __init__(self, block, layers, *, num_features=128, pyramid_levels=3, use_bn=True, k_bneck=1, k_upsample=3,
                 efficient=False, upsample_skip=True, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, detach_upsample_skips=(), detach_upsample_in=False,
                 align_corners=None, pyramid_subsample='bicubic', target_size=None,
                 output_stride=4, **kwargs):
        self.inplanes = 64
        self.efficient = efficient
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d if use_bn else Identity # Use batch_normalization flag (Used by default)
        # register_buffer is an inherited function from nn.module
        # It is used if you have parameters which should be saved and restored in the state_dict
        # but not trained by the optimizer
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1)) # view changes shape to 1x3x1x1        
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))   # I think thats to align the mean/std for each channel
        if scale != 1:                                                         # with normal image dimensions BxCxHxW
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.replicated = False

        self.align_corners = align_corners
        self.pyramid_subsample = pyramid_subsample
        self.bn1 = nn.ModuleList([bn_class(64) for _ in range(pyramid_levels)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        bottlenecks = []
        self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)] # self.inplanes = 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)] # self.inplanes = 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)] # self.inplanes = 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)] # self.inplanes = 512
        # Final bottlenecks list: [conv1x1(64, num_features), conv1x1(128, num_features), 
        #                          conv1x1(256, num_features), conv1x1(512, num_features)] (no downsampling)
        # BottleNecks are the red block in the diagram

        # log_2(4)=2 -> 2^2=4 
        num_bn_remove = max(0, int(log2(output_stride) - 2)) # simplified to max(0, 0) = 0
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove # 3 + 3 - 0 = 6
        bottlenecks = bottlenecks[num_bn_remove:] # currently not removing any bottlenecks

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.bn1]

        # []
        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1]) # [start:stop:step] - in this case it reverses the list

        num_pyr_modules = 2 + pyramid_levels - num_bn_remove # 2 + 3  = 5
        self.target_size = target_size
        if self.target_size is not None: # I think target_size is none by default
            h, w = target_size
            # Creates reverse list of ints (h, w) downsampled by powers of 2
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 2 + num_pyr_modules)][::-1]
        else:
            target_sizes = [None] * num_pyr_modules
        self.upsample_blends = nn.ModuleList(
            # Loops through target size list to grab the dimensions and pass into _UpsampleBlend
            [_UpsampleBlend(num_features,
                            use_bn=use_bn, # True
                            use_skip=upsample_skip, # True
                            detach_skip=i in detach_upsample_skips, # I think returns false each iteration bc detach is empty tuple 
                            fixed_size=ts,
                            k=k_upsample)
            for i, ts in enumerate(target_sizes)])
        self.detach_upsample_in = detach_upsample_in

        self.random_init = [self.upsample_bottlenecks, self.upsample_blends]

        self.features = num_features

        # Initialize tensors with certain values - I'm not sure exactly how this works
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        # used for layers 2-4 (downsampling is performed only at the beginning of these layers)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_class(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.efficient, bn_class=bn_class,
                            levels=self.pyramid_levels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_class=bn_class, levels=self.pyramid_levels, efficient=self.efficient))
        
        return nn.Sequential(*layers)

    def random_init_params(self):
        """Break down upsample_bottlenecks and upsample_blends layers' parameters into 1D list"""
        return chain(*[f.parameters() for f in self.random_init])
    
    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])
    
    def forward_resblock(self, x, layers, idx):
        skip = None
        for l in layers:
            x = l(x) if not isinstance(l, BasicBlock) else l(x, idx) # idx is used access the right bn layer
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image, skips, idx=-1):
        """Downsampling blocks of the architecture, this function is looped through in forward()
        for each branch (containing the original image or a downsampled image)
        
            Args:
                image: image or downsampled image to be convolved
                skips: Empty 2d list to store final ues
                idx: Index of the list of orginal and downsampled images
        """
        # Initial ResNet layers - 7x7 conv
        x = self.conv1(image)
        x = self.bn1[idx](x)
        x = self.relu(x)
        x = self.maxpool(x)

        # features list stores feature maps from each downsample block before the last relu
        # x = BasicBlock output w/ relu, skip = BasicBlock output w/o relu
        features = []
        x, skip = self.forward_resblock(x, self.layer1, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, idx)
        features += [skip]

        # Loop through the list of bottle necks and pass the features maps of the resnet
        # blocks to the 1x1 bottle necks (the red squares in the diagram).
        # A list of the new feature maps (skip_feats) is created in increasing resolution order
        #############figure out where concatenation occurs
        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        # skips value
        # Level 1: [[lay1_1] [lay1_2] [lay1_3] [lay1_4] [] []]
        # Level 2: [[lay1_1] [lay1_2,lay2_1] [lay1_3,lay2_2] [lay1_4,lay2_3] [lay2_4] []]
        # Level 3: [[lay1_1] [lay1_2,lay2_1] [lay1_3,lay2_2,lay3_1] [lay1_4,lay2_3,lay3_2] [lay2_4,lay3_3] [lay3_4]]
        # Print this part when running to understand it better###############
        for i, s in enumerate(reversed(skip_feats)): # 4 loops
            skips[idx+i] += [s]

        return skips


    def forward(self, image):
        if isinstance(self.bn1[0], nn.BatchNorm2d):
            if hasattr(self, 'img_scale'): # False -> scale = 1 -> no img_scale attribute
                image /= self.img_scale
            # Standardize data
            image -= self.img_mean
            image /= self.img_std
        pyramid = [image]

        # Loop twice and create a pyramid list that downsamples the original image by 1/2 and 1/4
        for l in range(1, self.pyramid_levels): # Loop starts at 1 bc 0 would not change the scale factor
            if self.target_size is not None:
                ts = list([si // 2 ** l for si in self.target_size])
                pyramid += [
                    F.interpolate(image, size=ts, mode=self.pyramid_subsample, align_corners=self.align_corners)]
            else:
                pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                          align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)] # create empty 2d list with 6 elements
        additional = {'pyramid': pyramid}
        # Loop through the orginal and downsampled images, passing the image, empty 2d list and pyramid index
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]
        if self.detach_upsample_in: # False
            x = x.detach()
        # Loop through upsample_blend layers and the reversed skips list. Pass x (smallest image size last resnet block)   
        # and the sum of skips tensors for each loop, then update x with the output of the upsample_blend layers
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk)) # sum represents the green blocks in the diagram, and inside blend it sums again with x

        # Returns upsampled feature map and a dictionary with a list of the various image sizes
        return x, additional # Final num_feature_maps=128, logits and Upsampling is finished in semseg.forward()
    
    # Once again im not sure if this is ever called
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                  unexpected_keys, error_msgs)
        for bn in self.bn1:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model
