"""Helper functions for the models directory"""

import torch
import torch.nn.functional as F
import warnings

from torch import nn as nn

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False) 
bathnorm_momentum = 0.01 / 2


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SeparableConv2d(nn.Module):
    """Creates a Depthwise Separable Convolution module - I think this was only used for experimental purposes"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class _BNReluConv(nn.Sequential):
    """ Creates layers  BatchNorm2D -> ReLU -> Conv2D ->  
    It looks like they are using bn > relu > conv2d. I am not sure why it is in this order.
    """
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super().__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d # False
        #warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class _UpsampleBlend(nn.Module):
    """ 
    module flow: upsample -> input + skip -> BN -> ReLU -> Conv2D_3x3 -> 
    """
    
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None,
                 k=3, separable = False):
                 super().__init__()
                 # bn -> relu -> conv2d
                 self.blend_conv  =_BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
                 self.use_skip = use_skip
                 self.detach_skip = detach_skip  # I think this is false each iteration
                 #warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
                 self.upsampling_method = upsample
                 if fixed_size is not None:
                        self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
                        #warnings.warn(f'Fixed upsample size', UserWarning)
    
    def forward(self, x, skip):
        if self.detach_skip: # False I think
            warnings.warn(f'Fixed upsample size')
            skip = skip.detach()
        skip_size = skip.size()[-2:] # Upsample x to size other branches feature maps (RB -> UP)
        x = self.upsampling_method(x, skip_size)
        #print(skip.shape)
        if self.use_skip:
            # Add upsampled feature map and other branch feature maps of same dimensions (just after the green + cirlce on diagram)
            x = x + skip 
        x = self.blend_conv.forward(x) # Feed into bn -> relu -> conv_3x3 
        #print(x.shape)
        return x