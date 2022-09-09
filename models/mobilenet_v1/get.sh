#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_FILE=mobilenet_v1_1.0_160

if [ ! -f "$SCRIPT_DIR/${MODEL_FILE}_frozen.pb" ]; then
    curl -L "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/$MODEL_FILE.tgz" | tar -C $SCRIPT_DIR -xz
else
    echo "Model already there: $MODEL_FILE"
fi
