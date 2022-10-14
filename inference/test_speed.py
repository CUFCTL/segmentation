"""Script for testing certain timing operartions

Orginally was investigating to see if torch.tensors find the max index 
faster than numpy.arrays
"""
import torch
import numpy as np
from time import perf_counter

print('loading numpy array...')
logits = np.load('test1.npy')
print(f'numpy array shape: {logits.shape}')
def warm_start(np_array):
    print('Performing warm start...')
    for i in range(5):
        np.argmax(logits, axis=1)

def warm_start_tensors(tensor_array):
    print('Performing warm start...')
    for i in range(100):
        torch.max(tensor_array, 1)


warm_start(logits)


start_t  = perf_counter()

pred_np = np.argmax(logits, axis=1)

end_t = perf_counter()

print(f'numpy total_time {(end_t-start_t)}')

logits_tens_test = torch.from_numpy(logits)

warm_start_tensors(logits_tens_test)

start_t = perf_counter()
logits_tensor = torch.from_numpy(logits)

_, pred_tensor = torch.max(logits_tensor, 1)
end_t = perf_counter()

print(f'pytorch total_time {(end_t-start_t)}')
