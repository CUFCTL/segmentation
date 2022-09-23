"""Load onnx file with ONNX Runtime and inference on a live video or an image.

The dimensions of the image/video to inference need to match the input dimensions of the onnx model.
If the dimensions differ, recreate the onnx model using the convert_to_onnx.py in the swiftnet directory and
set the dimensions to the intended image/video dimensions.

Make sure you have the ONNX Runtime GPU version installed 'onnxruntime-gpu'.

Set up:
    1. In the main() function change the 'model_path' variable to the location of the onnx file
    2. If you areinferencing on an image, change the 'image_path' path variable in the main function
    3. Set the 'device_id' variable to the desired GPU 
       - Run the 'nvidia-smi' command to see which GPUs are avaible and their device id's

Usage: python inference.py <mode>
    - <mode> is either 'video' or 'image' based on what you want to inference

"""
import argparse
import onnx
import onnxruntime as ort # make sure it is the gpu version or else it will be using the cpu
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from time import perf_counter

from labels import ColorizeLabels
from cityscapes import Cityscapes

parser = argparse.ArgumentParser()
# Mode can equal 'image' or 'video'
parser.add_argument('mode', type=str, help='Inference on an image file (mode=image) or video frames (mode=video).')

def preprocess(pil_img):
    """Preprocess an image for inference on SwiftNet. 
    Note:
        The only preprocessing that needs to be done
        is transforming the image to (B, C, H, W). SwiftNet does NOT normalize an image between 0-1, it only
        standardizes it with the mean and standard deviation of the dataset it was trained on. This does not have
        to be done in the preprocessing step because it is implemented in the forward() method, so to change these
        values a new onnx model would need to be created. This mean and standard deviation is calculated from the 
        raw dataset (meaning the values are not normalized). This is odd because most image machine learning problems 
        normalize first. The mean and standard deviation for cityscapes SwiftNet uses are 
        [73.15, 82.90, 72.3] and [47.67, 48.49, 47.73] respectively.
    """
    trans_img = np.array(pil_img, dtype=np.float32)
    trans_img = np.ascontiguousarray(np.transpose(trans_img, (2, 0, 1)))
    #img_mean = [73.15, 82.90, 72.3] # 73.15 / 255 = 0.2869, 82.60 / 255 = 0.3239 ... -> [0.2869, 0.3239, 0.2835]
    #img_std = [47.67, 48.49, 47.73]
    #raw_img -= img_mean
    #raw_img /= img_std
    trans_img = np.expand_dims(trans_img, axis=0) # reshapes from (C, H, W) to (B, C, H, W)

    return trans_img


def old_inference_video(session, frame):
    processed_img = preprocess(frame)

    # Get input data for ort session: {onnx input layer name: data to be inferenced}
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}
                 #ort_session.get_inputs()[1].name: target_size,
                 #ort_session.get_inputs()[2].name: image_size}
    ort_outs = session.run(None, ort_inputs)

    logits = ort_outs[0]
    pred = np.argmax(logits, axis=1).astype(np.uint32)
    return pred


def inference_video(session, frame):
    processed_img = preprocess(frame)
    X_ortvalue = ort.OrtValue.ortvalue_from_numpy(processed_img, 'cuda', 0)


    # Get input data for ort session: {onnx input layer name: data to be inferenced}
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}
                 #ort_session.get_inputs()[1].name: target_size,
                 #ort_session.get_inputs()[2].name: image_size}
    io_binding = session.io_binding()
    io_binding.bind_input('input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    io_binding.bind_output('output', 'cuda')


    session.run_with_iobinding(io_binding)

    ort_output = io_binding.get_outputs()[0]
    logits = ort_output.numpy()

    logits_tensor = torch.from_numpy(logits)
    _, pred = torch.max(logits_tensor, dim=1)
    return pred


def inference_image(session, image):
    start = perf_counter()
    processed_img = preprocess(image)
    X_ortvalue = ort.OrtValue.ortvalue_from_numpy(processed_img, 'cuda', 0)
    
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}
                 #ort_session.get_inputs()[1].name: target_size,
                 #ort_session.get_inputs()[2].name: image_size}

    io_binding = session.io_binding()
    io_binding.bind_input('input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    io_binding.bind_output('output', 'cuda')

    session.run_with_iobinding(io_binding)

    ort_output = io_binding.get_outputs()[0]
    logits = ort_output.numpy()

    #starting  = perf_counter() 

    #np.save('test1', logits)

    logits_tensor = torch.from_numpy(logits)
    _, pred = torch.max(logits_tensor, dim=1)
    #pred = np.argmax(logits, axis=1)
    
    #ending = perf_counter()
    #print(f'max time: {ending-starting}')

    return pred


def old_inference_image(session, image):
    start = perf_counter()
    processed_img = preprocess(image)
    
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}
                 #ort_session.get_inputs()[1].name: target_size,
                 #ort_session.get_inputs()[2].name: image_size}

    ort_outs = session.run(None, ort_inputs)

    logits = ort_outs[0]
    pred = np.argmax(logits, axis=1)#.astype(np.uint32)
    
    return pred


def warm_start(image, ort, num_inferences=5):
    print('Performing a warm start...')
    for i in range(num_inferences):
        dummy_output = inference_image(ort, image)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    args = parser.parse_args()

    # Initialize transforms
    print('Setting on ONNX Runtime session...')
    color_info = Cityscapes.color_info
    to_color = ColorizeLabels(color_info)

    model_path = 'models/model_best_single_input_h480_w640.onnx'
    #model_path = 'models/model_best_one_input.onnx'

    # Check that the ONNX model is valid
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    processor = ort.get_device()
    device_id = 0
    print(f'Using {processor} : {device_id}')
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': device_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    
    # Set up ORT
    ort_session = ort.InferenceSession(model_path, providers=providers)

    mode = args.mode 
    if mode == 'video':
        print('Inferencing on live video stream')
         # Set up opencv video stream
        vid = cv2.VideoCapture(0)
        
        start_t = 0
        while(True):
            
            _, frame = vid.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # IMPORTANT: do not forget to conver to RGB before inferencing
            
            k = cv2.waitKey(10)

            # Exit video stream by pressing 'q'
            if k == ord('q'):
                break
            prediction = inference_video(ort_session, frame_rgb)
            colored_pred = to_color(prediction)
            # Combine orginal and segmented image horizontally
            combined_image = np.concatenate((frame, colored_pred[0]), axis=1)
            end_t = perf_counter()
            fps = 1 / (end_t-start_t)
            start_t = end_t
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps_text = f'FPS: {fps:.2f}'# {fps}'

            cv2.putText(combined_image, fps_text, (5, 30), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', combined_image)
    elif mode == 'image':
        print('Inferencing on an image')
        #image_path = 'bonn_000023_000019_leftImg8bit.png'
        image_path = 'frame0.jpg'

        pil_img = Image.open(image_path)

        # Warm up the GPU in order to get accurate timings
        warm_start(pil_img, ort_session, num_inferences=5)

        start_t = perf_counter()

        prediction = inference_image(ort_session, pil_img)

        colored_pred = to_color(prediction)
        colored_img = Image.fromarray(colored_pred[0], 'RGB') # convert to PIL image
        end_t = perf_counter()

        fps = 1 / (end_t-start_t)
        print(f'FPS: {fps:.2f}')
        print(f'Total time: {end_t-start_t:.2}')
        
        colored_img.save(f'segmented_{image_path}')
        colored_img.show()
    else:
        print('Error: Not a valid command line argument.')


if __name__ == '__main__':
    main()
