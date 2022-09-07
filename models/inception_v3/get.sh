#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_FILE=inception_v3_2016_08_28_frozen.pb

if [ ! -f $SCRIPT_DIR/$MODEL_FILE ]; then
    curl -L "https://storage.googleapis.com/download.tensorflow.org/models/$MODEL_FILE.tar.gz" | tar -C $SCRIPT_DIR -xz
else
    echo "Model already there: $MODEL_FILE"
fi
