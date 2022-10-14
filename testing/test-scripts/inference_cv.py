"""Attempting to load onnx model in OpenCV - Couldn't get the model to load, might
try in the future with a different model
"""
import cv2

onnx_path = 'models/model_best_one_input.onnx'

opencv_net, test = cv2.dnn.readNetFromONNX(onnx_path)
print("OpenCV model was successfully read. Layer IDs: \n", opencv_net.getLayerNames())

# Still needs to be preprocessed 
test_img = 'bonn_000023_000019_leftImg8bit.png'

get_opencv_dnn_prediction(opencv_net, test_img)