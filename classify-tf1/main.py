import argparse
import os
import time
import numpy as np
import tensorflow as tf

MODEL_INFO = {
    'mobilenet_v1': {
        'imgSize': 160,
        'labelsFile': '../models/imagenet_labels.txt',
        'graphFile': '../models/mobilenet_v1/mobilenet_v1_1.0_160_frozen.pb',
        'inputLayer': 'input',
        'outputLayer': 'MobilenetV1/Predictions/Reshape_1',
    },
    'inception_v3': {
        'imgSize': 299,
        'labelsFile': '../models/inception_v3/imagenet_slim_labels.txt',
        'graphFile': '../models/inception_v3/inception_v3_2016_08_28_frozen.pb',
        'inputLayer': 'input',
        'outputLayer': 'InceptionV3/Predictions/Reshape_1',
    },
}

MODEL_NAME = 'mobilenet_v1'
#MODEL_NAME = 'inception_v3'

IMG_SIZE = MODEL_INFO[MODEL_NAME]['imgSize']
IMG_MEAN = 0.0

LABELS_FILE = MODEL_INFO[MODEL_NAME]['labelsFile']
GRAPH_FILE = MODEL_INFO[MODEL_NAME]['graphFile']
INPUT_LAYER = MODEL_INFO[MODEL_NAME]['inputLayer']
OUTPUT_LAYER = MODEL_INFO[MODEL_NAME]['outputLayer']

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
    parser.add_argument('-l', '--loops', help='', type=int, default=1)
    args = parser.parse_args()

    if not args.input:
        raise Exception('Input source is not specified')

    # Load images
    imgs = [] # {name, data}[]
    if os.path.isdir(args.input):
        for fn in os.listdir(args.input):
            img = _load_image(os.path.join(args.input, fn))
            imgs.append({'name': fn, 'data': img})
    else:
        img = _load_image(args.input)
        imgs.append({'name': args.input, 'data': img})

    labels = _load_labels()

    # Graph is a combination of model definition and trained weights
    graph = _load_graph()

    input = graph.get_operation_by_name('import/' + INPUT_LAYER).outputs[0]
    output = graph.get_operation_by_name('import/' + OUTPUT_LAYER).outputs[0]

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    #config.log_device_placement = True

    loops = 0
    total_images = 0
    total_elapsed = 0.0
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        while True:
            start_time = time.time()
            for img in imgs:
                results = sess.run(output, {input: img['data']})

                # Print most relevant results if not in looped moode
                if args.loops == 1:
                    results = np.squeeze(results)
                    print('\nImage: {}\nLabels:'.format(img['name']))
                    for i in results.argsort()[-RESULT_COUNT:][::-1]:
                        print(labels[i], results[i])

            loops += 1

            # The first loop is warming up, don't measure
            if loops > 1:
                elapsed = time.time() - start_time
                total_images += len(imgs)
                total_elapsed += elapsed
                fps = float(len(imgs)) / elapsed
                avg_fps = float(total_images) / total_elapsed
                print('Loop: {}, FPS: {:.2f}, Avg FPS: {:.2f}'.format(loops, fps, avg_fps))

            if args.loops == 0:
                continue
            if loops == args.loops:
                break


if __name__ == '__main__':
    _main()
