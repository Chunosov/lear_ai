# Image classification example

Based on standard TF [example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py).

Download model and labels if they are not already downloaded:

```bash
../models/inception_v3/get.sh
```

[Prepare environment](../README.md#prepare-python-3-8)

Run the example

```bash
python main.py ../samples/docbrown.jpg
```

## Run with GPU acceleration

https://www.tensorflow.org/install/pip

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive), version [11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal) is claimed to be compatible with TF 2.10:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

:exclamation: The package is for Ubuntu 20.04, can't install the package on Ubuntu 22.04, there is no package for 22 at the time of writing.

Install [cuDNN SDK 8.1.0](https://developer.nvidia.com/cudnn). There is no direct download link, nvidia decided that you must be a registerd user to be able to download the package :disappointed:. Even though it's freely available as part of their docker image `docker pull nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04`.

```bash
# Despite its name says "+cuda", it doesn't contain CUDA toolkit
sudo dpkg -i libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
```

## Run with GPU acceleration in docker

This approach uses a docker image already containig both CUDA Toolkit and cuDNN libraries.

```bash
../gpu-node/run.sh build

# in the container
cd classify-ft
python3 main.py ../samples/docbrown.jpg
```

## Clarification for some log messages

### `[W]` About TensorRT

```log
2022-09-08 14:00:33.502498: W tensorflow/stream_executor/platform/default/dso_loader.cc:64]
Could not load dynamic library 'libnvinfer.so.7';
dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory;

2022-09-08 14:00:33.502805: W tensorflow/stream_executor/platform/default/dso_loader.cc:64]
Could not load dynamic library 'libnvinfer_plugin.so.7';
dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory;

2022-09-08 14:00:33.502850: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]
TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
```

They refer to TensorRT which is optional and can be [safely ignored](https://stackoverflow.com/questions/60368298/could-not-load-dynamic-library-libnvinfer-so-6).

### `[I]` About NUMA node

```log
2022-09-08 14:14:46.772623: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980]
successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
```

This is not an error, can be ignored or can be [fixed by editing numa files](https://stackoverflow.com/questions/44232898/memoryerror-in-tensorflow-and-successful-numa-node-read-from-sysfs-had-negativ).

### `[E]` About cuBLAS

```log
2022-09-08 14:14:45.279095: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981]
Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

This message is even listed in an [official tutorial](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras) and seems nobody cares. So we don't either.
