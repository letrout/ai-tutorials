#!/bin/env python

"""
Based on https://hackernoon.com/introduction-of-tensorflow-with-python-f4a9624f2ab2
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import time

MAX_ITER_SEC = 10
MATRIX_MIN = 500
MATRIX_MAX = 50000
MATRIX_STEP = 50

# Put a leash on tf logging to console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def get_times(maximum_time):
    rows = list()
    device_names = ["/cpu:0"]
    if tf.test.is_gpu_available():
        device_names.append("/gpu:0")
    matrix_sizes = range(MATRIX_MIN,MATRIX_MAX,MATRIX_STEP)

    max_time = False
    for size in matrix_sizes:
        if max_time:
            break
        for device_name in device_names:

            shape = (size,size)
            data_type = tf.float16
            with tf.device(device_name):
                r1 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                r2 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                dot_operation = tf.matmul(r2, r1)

            print("####### Calculating on the " + device_name + " #######")
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                start_time = time.time()
                result = session.run(dot_operation)
                time_taken = time.time() - start_time
                #print(result)
                rows.append({'matrix': size, device_name: time_taken})

            if time_taken > maximum_time:
                max_time = True
                break
    columns = list(['matrix'])
    columns.extend(device_names)
    return pd.DataFrame(rows, columns=columns)

df = get_times(MAX_ITER_SEC)

matrix_sizes = df['matrix'].tolist()
for column in df.columns.tolist():
    if column == 'matrix':
        continue
    times = df[column].tolist()
    plt.plot(matrix_sizes, times, 'o-', label=column.strip('/'))
plt.ylabel('Time')
plt.xlabel('Matrix size')
plt.legend()
plt.show()