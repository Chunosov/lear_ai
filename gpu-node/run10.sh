#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

IMG=learn-ai-cuda10

if [ "$1" == "build" ]; then
    docker build -t $IMG -f Dockerfile-cuda10 .
    if [ $? -ne 0 ]; then exit 1; fi
fi

LIBS=/usr/lib/x86_64-linux-gnu

# This is the only needed to run the inference
LIB_CUDA=$LIBS/libcuda.so.1

docker run -it --rm \
    --device=/dev/nvidia0:/dev/nvidia0 \
    --device=/dev/nvidiactl:/dev/nvidiactl \
    --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
    --volume=$LIB_CUDA:$LIB_CUDA \
    --volume=$SCRIPT_DIR/..:/learn_ai \
    --workdir=/learn_ai/classify-tf1 \
    $IMG
