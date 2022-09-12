import argparse
import os
import time
import numpy as np
import cv2
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

LABELS_FILE = MODEL_INFO[MODEL_NAME]['labelsFile']
GRAPH_FILE = MODEL_INFO[MODEL_NAME]['graphFile']
INPUT_LAYER = MODEL_INFO[MODEL_NAME]['inputLayer']
OUTPUT_LAYER = MODEL_INFO[MODEL_NAME]['outputLayer']

RESULT_COUNT = 5

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

    loops = 0
    total_images = 0
    total_elapsed = 0.0
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        while True:
            start_time = time.time()
            for img in imgs:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: img['data']
                })

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
