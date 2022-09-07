import argparse
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 299
IMG_MEAN = 0.0

LABELS_FILE = '../models/inception_v3/imagenet_slim_labels.txt'
GRAPH_FILE = '../models/inception_v3/inception_v3_2016_08_28_frozen.pb'
INPUT_LAYER = "import/input"
OUTPUT_LAYER = "import/InceptionV3/Predictions/Reshape_1"

RESULT_COUNT = 5

# https://www.tensorflow.org/api_docs/python/tf/image/resize
def _load_image(file_name: str):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    img = img.reshape([1, IMG_SIZE, IMG_SIZE, 3])
    img = img.astype(np.float)
    # Normalize
    img = img / 255.0
    img = img - 0.5
    img = img * 2
    # Subtract mean
    img = img - np.mean(img)
    return img

    # This is in the original example
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
    # but doesn't work because seesion can't be run without a graph
    # Kept here as an alternative approach (but to be fixed)
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
    lines = tf.io.gfile.GFile(LABELS_FILE).readlines()
    return [l.rstrip() for l in lines]


def _load_graph():
    graph_def = tf.compat.v1.GraphDef()
    with open(GRAPH_FILE, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.compat.v1.Graph()
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

    input_operation = graph.get_operation_by_name(INPUT_LAYER)
    output_operation = graph.get_operation_by_name(OUTPUT_LAYER)

    with tf.compat.v1.Session(graph=graph) as sess:
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
