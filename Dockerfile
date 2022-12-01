FROM nvcr.io/nvidia/pytorch:21.08-py3
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY inference /app/inference 
COPY models /app/models
COPY scripts /app/scripts

# Required to run apt update (or at least it only worked for me)
USER root

# Dependencies for opencv
#RUN apt update
#RUN apt install ffmpeg libsm6 libxext6  -y

RUN pip install \
        onnx \
        onnxruntime-gpu \
        tqdm \
        cityscapesscripts 
        #opencv-python==4.4.*

WORKDIR "/app/"

CMD []