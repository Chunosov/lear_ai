import numpy as np
import tensorflow as tf
from datetime import datetime

# Show available devices
print("--" * 20)
with tf.compat.v1.Session() as sess:
    print("Devices:", sess.list_devices())

results = []

# Choose which device you want to test on: either 'cpu' or 'gpu'
for device in ['/device:CPU:0', '/device:XLA_CPU:0', '/device:GPU:0', '/device:XLA_GPU:0']:
    print("--" * 20)

    # Choose size of the matrix to be used.
    # Make it bigger to see bigger benefits of parallel computation
    for shape in [(50, 50), (100, 100), (500, 500), (1000, 1000)]:

        with tf.device(device):
            random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        start_time = datetime.now()
        with tf.compat.v1.Session(config=config) as sess:
            result = sess.run(sum_operation)
        elapsed = datetime.now() - start_time

        r = "Shape: {}, Device: {}, Time: {:.2f}".format(shape, device, elapsed.seconds + elapsed.microseconds/1e6)
        results.append(r)
        print(r)

print("--" * 20)
for r in results:
    print(r)
