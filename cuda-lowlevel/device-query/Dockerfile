FROM nvidia/cuda:10.0-devel-ubuntu18.04

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-samples-10-0 && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends cuda-samples-10-0

WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery
RUN make

CMD ./deviceQuery
