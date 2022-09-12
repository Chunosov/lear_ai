import argparse
import numpy as np
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
def _load_image(file_name):
    read_file = tf.io.read_file(file_name)
    decode_img = tf.io.decode_jpeg(read_file, channels=3)
    cast_float = tf.cast(decode_img, tf.float32)
    expand_dims = tf.expand_dims(cast_float, 0)
    resize = tf.image.resize(expand_dims, [IMG_SIZE, IMG_SIZE])
    subt_mean = tf.subtract(resize, [IMG_MEAN])
    normalize = tf.divide(subt_mean, [255.0])
    sess = tf.compat.v1.Session()
    return sess.run(normalize)


def _load_labels():
    with open(LABELS_FILE) as f:
        return [l.rstrip() for l in f.readlines()]


def _load_graph():
    graph_def = tf.compat.v1.GraphDef()
    with open(GRAPH_FILE, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def _main():
    print('Sample image classification')

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    args = parser.parse_args()

    if not args.input:
        raise Exception('Input source is not specified')

    img = _load_image(args.input)

    labels = _load_labels()

    # Graph is a combination of model definition and trained weights
    graph = _load_graph()

    input_operation = graph.get_operation_by_name('import/' + INPUT_LAYER)
    output_operation = graph.get_operation_by_name('import/' + OUTPUT_LAYER)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    #config.log_device_placement = True

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
