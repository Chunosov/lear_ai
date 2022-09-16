FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-10-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQueryDrv
RUN make

CMD ./deviceQueryDrv
