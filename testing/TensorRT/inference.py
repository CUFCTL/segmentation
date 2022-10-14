"""inference.py

Notes:
    1) During inference, smaller batch size -> prioritzes latency (time taken to process one unit of data)
                         larger batch size -> prioritizes throughput (number of data units that are processed in a specific period of time)
                         larger batch sizes take longer to process but reduce the average time spent on each sample
    2) TensorRT batch sizes should be static to make additional optimizations but it can handle dynamic batch sizes if needed
    3) The batch size is set during ONNX conversion
"""
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Normalize
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

N_CLASSES = 19
BATCH_SIZE = 1
PRECISION_1 = np.float16
PRECISION_2 = np.float32

file = open('models/model_best_bilinear.trt', 'rb')
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

engine = runtime.deserialize_cuda_engine(file.read())
context = engine.create_execution_context()

test_img = Image.open('bonn_000023_000019_leftImg8bit.png').convert('RGB')
np_img = np.array(test_img)

img_batch = np.array(np.expand_dims(np.array(np_img, dtype=np.float32), axis=0), dtype=np.float32) # import to convert to float, int is default
print(f"\nOriginal image batch shape and data type {img_batch.shape} {img_batch.dtype}")
# original shape => (1024, 2048, 3) => (H, W, C)
# transpose(0,2) => (3, 2048, 1024) => (C, W, H)
# transpose(1,2) => (3, 1024, 2048) => (C, H, W)
# np.expand_dims(new_img, axis=0 (1, 3, 1024, 2048) => (B, C, H, W)
def preprocess_image(img):
    #norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    result = torch.from_numpy(img).transpose(0,2).transpose(1,2)
    return np.array(result, dtype=np.float16)

final_img_batch= np.array([preprocess_image(image) for image in img_batch])

print(f"Preprocessed image batch shape {final_img_batch.shape} {final_img_batch.dtype}\n")

#dummy_input_batch = np.zeros((BATCH_SIZE, 1024, 2048, 3)) # np zeros shape is (B, H, W, D)


# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, N_CLASSES], dtype=PRECISION_1)

# Allocate device memory
d_input = cuda.mem_alloc(1 * img_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output

print("Warming up...")

pred = predict(final_img_batch)

print("Done warming up!")

pred = predict(final_img_batch)
print(pred)
