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

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
engine_file = "models/model_best_bilinear.trt"
input_file = "bonn_000023_000019_leftImg8bit.png"
output_file = "output.png"



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


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

final_img_batch= np.array([preprocess_image(image) for image in img_batch])

print(f"Preprocessed image batch shape {final_img_batch.shape} {final_img_batch.dtype}\n")




#dummy_input_batch = np.zeros((BATCH_SIZE, 1024, 2048, 3)) # np zeros shape is (B, H, W, D)


# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, N_CLASSES, ], dtype=PRECISION_1)

# Allocate device memory
d_input = cuda.mem_alloc(1 * img_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

def infer(engine, input_file, output_file):
    print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height


    #runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    #engine = runtime.deserialize_cuda_engine(file.read())
    #context = engine.create_execution_context()
    #test_img = Image.open('bonn_000023_000019_leftImg8bit.png').convert('RGB')
    #np_img = np.array(test_img)
    with engine.create_execution_context() as context:
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)#, None)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()

########################## Start Here ##########################################
################# Working through this tutorial 
# ############### https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb

def predict(batch): # result gets copied into output
    # transfer input data to device
    # execute model
    # transfer predictions back
    # syncronize threads
    stream.synchronize()
    
    return output

print("Warming up...")

pred = predict(final_img_batch)

print("Done warming up!")

pred = predict(final_img_batch)
print(pred)
