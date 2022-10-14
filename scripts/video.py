import numpy as np
import cv2
from PIL import Image
import torch
import torchvision

def opencv_video():
    vid = cv2.VideoCapture(0)
    count = 0
    while(True):

        # Capture the video frame
        _, frame = vid.read()
        h, w, d = frame.shape
        #combined_image = np.zeros((h, w*2, d), np.uint8)
        #print(combined_image.shape)
        #combined_image[:h, :w] = frame
        #combined_image[:h, w:] = frame

        # Alternative concatenation - horizontal stacking
        combined_image_np = np.concatenate((frame, frame), axis=1) # axis=0 => vertical image stacking        
        # Display the frame
        cv2.imshow('frame', combined_image_np)

        # Wait for key press for 10ms and return the keys value
        k = cv2.waitKey(10) # 1ms is too fast and does not always register key press

        if  k == ord('s'): # 's' save video frame
            cv2.imwrite(f"/home/eceftl7/temp/frame{count}.jpg", frame) # Save image using opencv
            
            # Convert cv image to PIL image
            pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv uses bgr format instead of rgb so that must be converted
            pil_frame = Image.fromarray(pil_frame)
            pil_frame.save(f"/home/eceftl7/temp/frame{count}_PIL.jpg") # Save image using pil
            pil_frame_test = pil_frame.convert('RGB')
            print(count)
            print(pil_frame.size)

            count += 1 

        elif k == ord('q'): # 'q'
            break

    vid.release()
    cv2.destroyAllWindows()


def torch_video():
    stream = 'video'

    video_path = '/dev/video0'
    video = torchvision.io.VideoReader(video_path, stream)
    video.get_metadata()


def main():
    opencv_video()
    #torch_video()




if __name__ == '__main__':
    main()
