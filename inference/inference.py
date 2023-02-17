"""Load onnx file with ONNX Runtime and inference on a live video or an image.

The dimensions of the image/video to inference need to match the input dimensions of the onnx model.
If the dimensions differ, recreate the onnx model using the convert_to_onnx.py in the swiftnet directory and
set the dimensions to the intended image/video dimensions.

Make sure you have the ONNX Runtime GPU version installed 'onnxruntime-gpu'.

Setup:
    1. In the main() function change the 'model_path' variable to the location of the onnx file
    2. If you are inferencing on an image, change the 'image_path' path variable in the main function
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
import time

from utils import color_maps
from utils import labels


parser = argparse.ArgumentParser()
parser.add_argument('--mode',
    default='video',
    choices=['video', 'image'],
    type=str,
    help='Inference on an image file <mode>=image or video frames <mode>=video.'
)
parser.add_argument('--onnx_file',
    default='models/model_cityscapes.onnx',
    type=str, 
    help='Path to onnx file. Ex: models/model_cityscapes.onnx',
    metavar='' # Helps allign the -h output (puts a blank '' in the example)
)
parser.add_argument('-r',
    action='store_true', 
    help="""Optional flag to inference on a recorded video instead of a live video 
            If the flag is provided, the video in the 'video_path' variable is loaded'"""
)
parser.add_argument('--image_path',
    default='media/bonn_000023_000019_leftImg8bit.png',
    type=str,
    help="""Path to video file (probably mp4). Only required if the -r flag is specified. 
            Ex: media/go-pro-1.mp4""",
    metavar=''
)
parser.add_argument('--video_path', 
    default='media/go-pro-1.MP4',
    type=str,
    help="""Path to video file (probalby mp4). Only required if the -r flag is specified. 
            Ex: media/go-pro-1.mp4""",
    metavar=''
)
parser.add_argument('--cmap',
    default='cityscapes',
    choices=['cityscapes', 'rellis'],
    type=str,
    help='Color map to be applied to the prediction',
    metavar=''
)

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
    trans_img = np.ascontiguousarray(np.transpose(trans_img, (2, 0, 1))) # reshapes from (H, W, C) to (C, H, W)
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
    ort_outs = session.run(None, ort_inputs)

    logits = ort_outs[0]
    pred = np.argmax(logits, axis=1).astype(np.uint32)
    return pred


def inference_video(session, frame, output_dimensions):
    processed_img = preprocess(frame)
    X_ortvalue = ort.OrtValue.ortvalue_from_numpy(processed_img, 'cuda', 0)


    # Get input data for ort session: {onnx input layer name: data to be inferenced}
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}
    io_binding = session.io_binding()
    io_binding.bind_input('input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    
    
    # Allocate GPU tensor so we can perform argmax on GPU
    # (This did not speed it up much at all but it is more logical)
    Y_shape = (1, output_dimensions[2], output_dimensions[0], output_dimensions[1])
    Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
    io_binding.bind_output(
        name='output',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(Y_tensor.shape),
        buffer_ptr=Y_tensor.data_ptr(),
    )

    session.run_with_iobinding(io_binding)

    _, pred = torch.max(Y_tensor, dim=1)
    return pred


def inference_image(session, image, output_dimensions):
    start = time.time()
    processed_img = preprocess(image)
    X_ortvalue = ort.OrtValue.ortvalue_from_numpy(processed_img, 'cuda', 0)
    
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}

    io_binding = session.io_binding()
    io_binding.bind_input('input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    Y_shape = (1, output_dimensions[2], output_dimensions[0], output_dimensions[1])
    Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
    io_binding.bind_output(
        name='output',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(Y_tensor.shape),
        buffer_ptr=Y_tensor.data_ptr(),
    )

    session.run_with_iobinding(io_binding)

    _, pred = torch.max(Y_tensor, dim=1)
    return pred
    


def old_inference_image(session, image):
    """Old method of ort inference that is slower than the new method"""
    start = time.time()
    processed_img = preprocess(image)
    
    ort_inputs = {session.get_inputs()[0].name: np.array(processed_img)}

    ort_outs = session.run(None, ort_inputs)

    logits = ort_outs[0]
    pred = np.argmax(logits, axis=1)#.astype(np.uint32)
    
    return pred


def warm_start(image, ort, num_inferences=5):
    print('Performing a warm start...')
    for i in range(num_inferences):
        _ = inference_image(ort, image)


def main():
    args = parser.parse_args()

    # Initialize transforms
    print('Setting on ONNX Runtime session...')

    # Get color map and convert to BGR (OpenCV requirement)
    cmap = np.array(color_maps[args.cmap])
    label_names = np.array(labels[args.cmap])
    cmap_bgr = cmap[:,::-1].astype(np.uint8).flatten().tolist()

    model_path = args.onnx_file

    # Check that the ONNX model is valid
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    processor = ort.get_device()
    device_id = 0
    print(f'\nUsing {processor} : {device_id}')
    if processor == 'GPU':
        print(f'{torch.cuda.get_device_name(device_id)}\n')
    
    providers = [
        ('CUDAExecutionProvider', { # using default dict params from ONNX documentation
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
        # Set up opencv video stream
        video_path = args.video_path
        if args.r:
            print('Inferencing on a recorded video')
            print(video_path)
            vid = cv2.VideoCapture(video_path)
        else:
            print('Inferencing on live video stream')
            vid = cv2.VideoCapture(0)

        check, frame = vid.read()

        H, W, C = frame.shape

        legend = np.full((H, 120, C), 255).astype(np.uint8)

        # Create legend for final visualization
        (x1, y1), (x2, y2) = (5, 30), (40, 50)
        for color, label in zip(cmap, label_names):
            label_colors = (int(color[2]), int(color[1]), int(color[0]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(legend, (x1, y1), (x2, y2), label_colors, -1)
            cv2.putText(legend, label, (x1+36, y1+13), font, .5, (0, 0, 0), 0, cv2.LINE_AA)
            y1 += 20
            y2 += 20

        # Define output shape
        out_C = len(cmap)
        out_dimensions = (H, W, out_C)

        start_t = 0
        while(True):
            check, frame = vid.read()
            # Check if frame exists before processing - some of the Go Pro video frames were corrupted
            if not check:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # IMPORTANT: do not forget to convert to RGB before inferencing
            
            k = cv2.waitKey(10)

            # Exit video stream by pressing 'q'
            if k == ord('q'):
                break
            prediction = inference_video(ort_session, frame_rgb, out_dimensions)
            # squeeze() removes all axes with length of 1 ... [1, 1080, 1920] => [1080, 1920]
            # Image modes ('P') defined here: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            prediction_pil = Image.fromarray(prediction.cpu().numpy().squeeze().astype(np.uint8)).convert('P')
            prediction_pil.putpalette(cmap_bgr)
            # Combine orginal and segmented image horizontally
            colored_pred = prediction_pil.convert('RGB')
            combined_image = np.concatenate((frame, colored_pred, legend), axis=1)

            end_t = time.time()
            fps = 1 / (end_t-start_t)
            start_t = end_t
            fps_text = f'FPS: {fps:.2f}'# {fps}'

            # BGR
            box_color = (27, 122, 251)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(combined_image, (4, 2), (180,35), box_color, -1)
            cv2.putText(combined_image, fps_text, (5, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', combined_image)
    elif mode == 'image':
        print('Inferencing on an image')
        image_path = args.image_path

        pil_img = Image.open(image_path)

        # Warm up the GPU in order to get accurate timings
        warm_start(pil_img, ort_session, num_inferences=5)

        start_t = time.time()

        prediction = inference_video(ort_session, frame_rgb)
        # squeeze() removes all axes with length of 1 ... [1, 1080, 1920] => [1080, 1920]
        # Image modes ('P') defined here: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        prediction_pil = Image.fromarray(prediction.numpy().squeeze().astype(np.uint8)).convert('P')
        prediction_pil.putpalette(cmap_bgr)
        # Combine orginal and segmented image horizontally
        colored_pred = prediction_pil.convert('RGB')
        end_t = time.time()

        fps = 1 / (end_t-start_t)
        print(f'FPS: {fps:.2f}')
        print(f'Total time: {end_t-start_t:.2}')
        
        image_name = image_path.split('/')[-1]
        prediction_pil.save(f'media/segmented_{image_name}')
        prediction_pil.show()
    else:
        print('Error: Not a valid command line argument.')


if __name__ == '__main__':
    main()
