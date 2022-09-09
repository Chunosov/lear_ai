import argparse
import numpy as np
import cv2
import tensorflow as tf

#IMG_SIZE = 299 # ../models/inception_v3
IMG_SIZE = 160 # ../models/mobilenet_v1
IMG_MEAN = 0.0

LABELS_FILE = '../models/imagenet_labels.txt'
#GRAPH_FILE = '../models/inception_v3/inception_v3_2016_08_28_frozen.pb'
GRAPH_FILE = '../models/mobilenet_v1/mobilenet_v1_1.0_160_frozen.pb'
INPUT_LAYER = 'input'
#OUTPUT_LAYER = 'InceptionV3/Predictions/Reshape_1'
OUTPUT_LAYER = 'MobilenetV1/Predictions/Reshape_1'

RESULT_COUNT = 5

# https://www.tensorflow.org/api_docs/python/tf/image/resize
def _load_image(file_name: str):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    img = img.reshape([1, IMG_SIZE, IMG_SIZE, 3])
    img = img.astype(float)
    # Normalize
    img = img / 255.0
    img = img - 0.5
    img = img * 2
    # Subtract mean
    img = img - np.mean(img)
    return img

    # This is in the original example
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
    # but doesn't work because session can't be run without a graph
    # Kept here as an alternative approach
    #
    # read_file = tf.io.read_file(file_name)
    # decode_img = tf.io.decode_jpeg(read_file, channels=3)
    # cast_float = tf.cast(decode_img, tf.float32)
    # expand_dims = tf.expand_dims(cast_float, 0)
    # resize = tf.image.resize(expand_dims, [IMG_SIZE, IMG_SIZE])
    # subt_mean = tf.subtract(resize, [IMG_MEAN])
    # normalize = tf.divide(subt_mean, [255.0])
    # sess = tf.compat.v1.Session()
    # return sess.run(normalize)


def _load_labels():
    with open(LABELS_FILE) as f:
        return [l.rstrip() for l in f.readlines()]
    #lines = tf.io.gfile.GFile(LABELS_FILE).readlines()
    #return [l.rstrip() for l in lines]


def _load_graph():
    graph_def = tf.compat.v1.GraphDef()
    with open(GRAPH_FILE, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def _main():
    print('-------------------------------------------')
    print('Sample image classification')

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    args = parser.parse_args()

    if not args.input:
        raise Exception('Input source is not specified')

    img = _load_image(args.input)

    labels = _load_labels()

    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_visible_devices(gpus[0], 'GPU')

        # mem_cfg = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024+512)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [mem_cfg])

        # mem_cfg = tf.config.LogicalDeviceConfiguration(memory_limit=1024+512)
        # tf.config.set_logical_device_configuration(gpus[0], [mem_cfg])

    print('-------------------------------------------')


    # Graph is a combination of model definition and trained weights
    graph = _load_graph()

    input_operation = graph.get_operation_by_name('import/' + INPUT_LAYER)
    output_operation = graph.get_operation_by_name('import/' + OUTPUT_LAYER)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: img
        })
    results = np.squeeze(results)

    # Print most relevant results
    print('\nLabels:')
    for i in results.argsort()[-RESULT_COUNT:][::-1]:
        print(labels[i], results[i])


if __name__ == '__main__':
    _main()
