# I think this docker image only supports GPUs with 6.0 compute capability 
# and Quadro M1200 is only 5.0 and cannot find an image with 5.0 compute capability
FROM nvcr.io/nvidia/pytorch:21.08-py3
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Required to run apt update (or at least it only worked for me)
#USER root

# Dependencies for opencv
RUN apt update 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install ffmpeg libsm6 libxext6 -y

RUN pip install \
        onnx \
        onnxruntime-gpu \
        tqdm \
        cityscapesscripts \
        opencv-python==4.1.2.30
        #opencv-python==4.4.0.40

# Copying source files after installing dependcies to prevent long rebuilds
COPY inference /app/inference 
COPY models /app/models
COPY scripts /app/scripts

WORKDIR "/app/"

CMD []