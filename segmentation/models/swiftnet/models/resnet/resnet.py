"""ResNet modified to use atrous convolutions. Originally made for DeepLab

Code taken from: https://github.com/bradford415/deeplabv3-pytorch/blob/main/deeplabv3.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# List of backbones
# Double underscore is just naming convention, 
# allows you to use all as a variable and ignores the 'all' keyword/function
__all___ = ['resnet101'] 

class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        """
        Args:
            C: number of in_channels
            depth: number of out_channels
        """
        super().__init__()
        self._C = C
        self._depth = depth
        self._num_classes=num_classes
        # Global average pooling is used to help the problem of losing information
        # when atrous rates are large enough, close to the size of the feature map.
        # This is explained in the deeplabv3 paper section 3.3. This is  performed 
        # in the x5 step
        self.global_pooling = nn.AdaptiveAvgPool2d(1) # Global pooling, output size 1
        self.relu = nn.ReLU(inplace=True)
        # Defining convolutions with atrous rates [6, 12, 18]
        self.aspp1 = conv(C,depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1, # padding prevents downsampling
                          dilation=int(6*mult), padding=int(6*mult), bias=False) 
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(12*mult), padding=int(12*mult), bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(18*mult), padding=int(18*mult), bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth*5, depth, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)
    
    def forward(self, x):
        # perform 5 different convolutions, each using the orignal data x
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        # x5 is used to incorporate global context information
        # AdapativeAvgPooling(1) returns only 1 pixel from each feature map 
        # so they need to be upsampled to the original feature map size of x
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        # Upsampling to orignal feature map (x) size
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', 
                         align_corners=True)(x5) # shape[2]=height, shape[3]=width
        # Concatenate feature maps from each convolution.
        # This will append feature maps on the 1st dimension which effectively 'stacks'
        # them like normal Conv2d with multiple output channels.
        # The feature maps are of size [batch_size, num_channels, height, width]
        # so dim=1 will append the feature maps to the num_channels dimension
        x = torch.cat((x1, x2, x3, x4, x5), dim=1) 
        # Fuse concatenated feature maps using 1x1 convolutions
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Generate the final logits using 1x1 convolution creating
        # the # of output channels = # of classes
        x = self.conv3(x)
        
        return x

class BottleNeck(nn.Module):
    """Modified BuildingBlock for deeper ResNets (50, 101, 152).
    Each BottleNeck has 3 layers instead of 2. The 3 layers are 1x1, 3x3, and 1x1 convolutions.
    The 1x1 layers are responsible for reducing and then increasing (restoring) dimensions
    
    """

    # Value to multiply the output channels by in the last 1x1 BottleNeck layer (Fig. 5 & Table 1)
    expansion = 4  

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=nn.Conv2d, norm=nn.BatchNorm2d):
        super().__init__()
        # As explained before, ResNet1.5 downsamples at self.conv3 and 
        # self.downsample when stride > 1
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        # BottleNeck is formed here bc conv1 has fewer output channels (planes) than input channels (inplanes)
        # Similarily, Conv3 takes fewer input channels and output greater channels.
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        # 3rd BottleNeck layer multiply out_channels by 4
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Perform a normal forward pass, then add the original input to the output
        of the convolutions. Then pass this final value to ReLU activation.
        """
        identity = x

        out = self.conv1(x) # uses 'out' variable bc you do not want to modify x
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # If downsampling also downsample the original input
        # This is so you are adding matrices of the same size  
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model modified for deeplab"""
    
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64 # PyTorch refers to planes as the number of channels for some reason
        self.dilation = 1
        self.conv = nn.Conv2d # Mostly use for 1x1 convolutions
        self.norm = nn.BatchNorm2d
        '''
        if replace_stride_with_dilation is None:
            # Each element in the list indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            # for deeplab you want the second and third element to be true.
            # This enables block 3 and 4 to use atrous convolution.
            # For the normal resnet without dilation they all should be false
            # I think. This should be set when calling ResNet.
            replace_stride_with_dilation = [False, True, True]
        '''
        # No bias because batch normalization, next layer, takes care of it
        # Padding and stride values set of the first conv layer set by ResNet paper
        # This first layer downsamples the input by 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False) # [(W−K+2P)/S]+1 = Z
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # inplace can save a small amount of memory by not creating a new object
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
        #                       padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0]) # defining conv blocks after the first conv layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
         # Apply dilation only for block 4. If replace_stride_with_dilation[i]=True,
         # _make_layer() will set to stride to 1 and dilation will equal the stride argument value, this case 2
        self.aspp = ASPP(512*block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """Create convolutional blocks w/ hyperparameters specified by resnet.
        These conv blocks start after the first conv and max pooling layer.
        Args:
            block: The type of block to add. ResNet-18/34 uses 'BasicBlock' (BuidlingBlock
                    in ResNet paper) and ResNet-50/101/152 uses 'BottleNeck' block
            planes: Refers to the number of channels, (I'm not sure why they call them planes).
                    Inplanes is the number of input channels to the conv layer I think.
            num_blocks: The number of blocks to add per 'building', this will depend on the
                        type of resnetxxx (ex. 'resnet101' uses [3, 4, 23, 3]). these values
                        can be found in Table 1 in the resnet paper.
        """
        downsample = None
        # previous_dilation = self.dilation 
        # If you are downsampling the feature map. 
        # This happens at the end of the the first BottleNeck for each group of 
        # BottleNeck blocks (every time _makelayer is called except for the first one)
        # The number of BottleNecks is specified by the layers[] list in __init__().
        if dilation !=1 or stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes*block.expansion, # Cannot dilate a 1x1 convolution
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion)
            )   
        
        """From the original ResNet paper, after the block of 64 kernels, starting at 128, 
        the first BottleNeck in a group downsamples during the 1st 1x1 Conv and at the end of 
        the BottleNeck, every block after will not. We start the for loop at 1 to skip this 
        first downsampling layer we already appended. According to a new paper, 
        https://arxiv.org/abs/1512.03385, downsampling, w/ stride > 1, in the BottleNeck 
        during the first 3x3 convolution (instead of the first 1x1) improves accuracy and is what 
        the PyTorch github implements. This variant is called ResNet V1.5. Similarily, they 
        downsample in the 'BasicBlock' on first 3x3 conv layer of each group of blocks, but not 
        the second 3x3 conv (because BasicBlock contains two 3x3 Convs)
        """
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, 
                          conv=self.conv, norm=self.norm))
        return nn.Sequential(*layers) 
        # * operator expands the list into positional arguments to create a model in 1 line
    
    def forward(self, x):
        size = (x.shape[2], x.shape[3]) # get original image height
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)

        return x

def create_resnet101(pretrained=False, device='cpu', **kwargs):
    """ Contstruct a ResNet-101 model
    
    Args:
        pretrained (bool): If True, load pre-trained ImageNet weights
    """
    model = ResNet(BottleNeck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # A model's state dictionary maps each layer to its parameter tensor
        # Only layers that have learnable parameters have entries in the dictionary
        model_dict = model.state_dict()
        # Download pre-trained resnet101 model from PyTorch and get its state_dict()
        resnet101 = models.resnet101(pretrained=True)
        pretrained_dict = resnet101.state_dict()
        # Filter out unncessary keys - only return parameters from the pre-trained
        # ResNet that matches our ResNet 
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # might not be necessary, maybe can just pass overlay_dict to load_state_dict()
        model_dict.update(overlap_dict) 
        model.load_state_dict(model_dict) # Load pre-trained weights
        # torch.load did not work and im not sure of another way to map location 

    return 