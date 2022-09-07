#!/bin/bash

CMD=$1

RTSP=rtsp://localhost:8554

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


if [ "$CMD" == "server" ]; then
    # Download and run rtsp-simple-server
    # https://github.com/aler9/rtsp-simple-server

    cd $SCRIPT_DIR

    if [ ! -f rtsp-simple-server ]; then
        PKG=rtsp-simple-server_v0.20.0_linux_amd64.tar.gz
        if [ ! -f $PKG ]; then
            wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.20.0/$PKG
        fi
        tar -zxvf $PKG
    fi

    export RTSP_PROTOCOLS=tcp
    ./rtsp-simple-server
    exit 0
fi

if [ "$CMD" == "stream" ]; then
    # Stream video (https://trac.ffmpeg.org/wiki/StreamingGuide) file to the local RTSP server

    FILE=$2
    CHAN=$3
    ADDR=$RTSP/$CHAN
    echo "Streaming $FILE to $ADDR ..."
    ffmpeg -re -stream_loop -1 -i $FILE -c copy -f rtsp -rtsp_transport tcp $ADDR
    exit 0
fi

if [ "$CMD" == "show" ]; then
    # Show and RTSP stream from local RTSP server using VLC

    CHAN=$2
    ADDR=$RTSP/$CHAN
    echo "Opening $ADDR ..."

    # VLC seems broken on newer kernels. Can't use it on Ubuntu 22.04 kernel 5.15.0-47
    # Something similar to mentioned in https://github.com/aler9/rtsp-simple-server/issues/223
    # and in https://github.com/SvenVD/rpisurv/issues/136
    #vlc $ADDR

    # sudo apt install mplayer
    mplayer $ADDR
    exit 0
fi

echo "Command not specified"
exit 1
