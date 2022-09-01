import numpy as np
import torch
from PIL import Image

test_img = Image.open('bonn_000023_000019_leftImg8bit.png').convert('RGB')
label = Image.open('aachen_000000_000019_gtFine_labelTrainIds.png')

np_img = np.array(test_img)

print(np_img.shape) 
test = np.zeros((1,4,2,3))
print(np.array(label).shape[0:2])

# original shape => (1024, 2048, 3) => (H, W, C)
# transpose(0,2) => (3, 2048, 1024) => (C, W, H)
# transpose(1,2) => (3, 1024, 2048) => (C, H, W)
# np.expand_dims(new_img, axis=0 (1, 3, 1024, 2048) => (B, C, H, W)
new_img = torch.from_numpy(np_img).transpose(0,2).transpose(1,2)
final_img = np.expand_dims(new_img, axis=0)
print(final_img.shape)