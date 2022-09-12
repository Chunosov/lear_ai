# Object Detection Example

It uses [TensorFlow Lite](https://www.tensorflow.org/lite) and its [Python bindings](https://www.tensorflow.org/lite/guide/python) and [SSD Mobilenet V1 Model](https://iq.opengenus.org/ssd-mobilenet-v1-architecture/) trained for [COCO](https://cocodataset.org/#home) dataset. There is a [useful article](https://towardsdatascience.com/using-tensorflow-lite-for-object-detection-2a0283f94aed) giving an overall understanding of how the app works.

## Run

[Prepare environment](../README.md#prepare-python-3-8)

### Process single image

Detect objects in an image and show results in an OpenCV window:

```bash
python main.py ../samples/docbrown.jpg
```

or save result into another image with detection boxes overlayed:

```bash
python main.py ../samples/docbrown.jpg -o tmp.jpg
```

### Process RTSP stream

Run RTSP server with sample video file:

```bash
# Start server in a separate terminal
../rtsp.sh server

# Start streaming in a separate terminal
../rtsp.sh stream ../samples/traffic.ts ch1

# Check if all works with (optional)
../rtsp.sh show ch1
```

Detect an RTSP stream, it produces OpenCV video window with detection boxes overlayed:

```bash
python main.py rtsp://localhost:8554/ch1
```

Detect an RTSP stream, it produces another RTSP stream (ffmpeg required):

```bash
python main.py rtsp://localhost:8554/ch1 -o rtsp://localhost:8554/ch1-det

# and view results in a separate terminal
../rtsp.sh show ch1-det
```
