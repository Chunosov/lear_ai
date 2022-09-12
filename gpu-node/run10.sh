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

# These are not required for TF to run but are useful
# for checking if GPU is available in running container
LIB_NVML=$LIBS/libnvidia-ml.so.1
NVSMI=/usr/bin/nvidia-smi

docker run -it --rm --privileged \
    -v $LIB_CUDA:$LIB_CUDA \
    -v $LIB_NVML:$LIB_NVML \
    -v $NVSMI:$NVSMI \
    -v $SCRIPT_DIR/..:/learn_ai \
    -w /learn_ai/classify-tf1 \
    $IMG
