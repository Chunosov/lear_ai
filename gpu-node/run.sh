#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

IMG=learn-ai-cuda

if [ "$1" == "build" ]; then
    docker build -t $IMG .
    if [ $? -ne 0 ]; then exit 1; fi
fi

LIBS=/usr/lib/x86_64-linux-gnu

LIB_CUDA=$LIBS/libcuda.so.1
LIB_NVJIT=$LIBS/libnvidia-ptxjitcompiler.so.1

# These are not required for TF to run but are useful
# for checking if GPU is available in running container
LIB_NVML=$LIBS/libnvidia-ml.so.1
NVSMI=/usr/bin/nvidia-smi

docker run -it --rm --privileged \
    -v $LIB_CUDA:$LIB_CUDA \
    -v $LIB_NVML:$LIB_NVML \
    -v $LIB_NVJIT:$LIB_NVJIT \
    -v $NVSMI:$NVSMI \
    -v $SCRIPT_DIR/..:/learn_ai \
    -w /learn_ai \
    $IMG
