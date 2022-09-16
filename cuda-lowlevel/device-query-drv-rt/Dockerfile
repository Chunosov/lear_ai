FROM nvidia/cuda:10.0-devel-ubuntu18.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-10-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQueryDrv
RUN make

# Use only runtime image

FROM nvidia/cuda:10.0-runtime-ubuntu18.04

WORKDIR /
COPY --from=builder /usr/local/cuda/samples/1_Utilities/deviceQueryDrv/deviceQueryDrv .
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat
CMD ./deviceQueryDrv
