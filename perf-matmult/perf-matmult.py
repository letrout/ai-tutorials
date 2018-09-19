#!/bin/env python

"""
Based on https://hackernoon.com/introduction-of-tensorflow-with-python-f4a9624f2ab2
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

# Put a leash on tf logging to console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def get_times(maximum_time):

    device_times = {
        "/cpu:0":[]
    }
    if tf.test.is_gpu_available():
        device_times["/gpu:0"] = []
    matrix_sizes = range(500,50000,50)

    for size in matrix_sizes:
        for device_name in device_times.keys():

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
                device_times[device_name].append(time_taken)

            print(device_times)

            if time_taken > maximum_time:
                return device_times, matrix_sizes

device_times, matrix_sizes = get_times(10)

for device, times in device_times.items():
    plt.plot(matrix_sizes[:len(times)], times, 'o-')
plt.ylabel('Time')
plt.xlabel('Matrix size')
plt.show()