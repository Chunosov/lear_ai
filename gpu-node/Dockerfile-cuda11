FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

RUN apt update
RUN apt install -y file iputils-ping python3 python3-pip

# opencv also installs numpy
RUN pip install opencv-python==4.6.0.66
RUN pip install tensorflow==2.10.0

# these needed to make opencv working
RUN apt install -y libgl1
# suppress asking for current timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y libglib2.0-0


CMD bash
