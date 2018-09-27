#!/bin/env python

"""
Based on https://hackernoon.com/introduction-of-tensorflow-with-python-f4a9624f2ab2
"""

from __future__ import print_function
import argparse
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

def get_times(max_time, matrix_sizes, device_names):
    rows = list()
    max_hit = False
    for size in matrix_sizes:
        if max_hit:
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

            if time_taken > max_time:
                max_hit = True
                break
    columns = list(['matrix'])
    columns.extend(device_names)
    return pd.DataFrame(rows, columns=columns)

def parse_arguments():
    """
    Parse script arguments
    :return: argparse parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_time',
        dest='max_time',
        help="""Max run time in seconds (default={}""".format(MAX_ITER_SEC),
        type=int,
        default=MAX_ITER_SEC
    )
    parser.add_argument(
        '--matrix_min',
        dest='matrix_min',
        help="""Minimum matrix size (default={}""".format(MATRIX_MIN),
        type=int,
        default=MATRIX_MIN
    )
    parser.add_argument(
        '--matrix_max',
        dest='matrix_max',
        help="""Maximum matrix size (default={}""".format(MATRIX_MAX),
        type=int,
        default=MATRIX_MAX
    )
    parser.add_argument(
        '--matrix_step',
        dest='matrix_step',
        help="""Matrix step size (default={}""".format(MATRIX_STEP),
        type=int,
        default=MATRIX_STEP
    )
    parser.add_argument(
        '--no_cpu',
        dest='no_cpu',
        help='Do not run on CPU (default=False)',
        action='store_true'
    )
    parser.add_argument(
        '--no_gpu',
        dest='no_gpu',
        help='Do not run on GPU (default=False)',
        action='store_true',
    )
    parser.add_argument(
        '-d',
        '--directory',
        help='The directory in which to write csv file (default=None)',
        type=str,
        default=None
    )
    return parser.parse_args()

args = parse_arguments()
matrix_sizes = range(args.matrix_min,args.matrix_max,args.matrix_step)
device_names = list()
if not args.no_cpu:
    device_names.append("/cpu:0")
if not args.no_gpu and tf.test.is_gpu_available():
    device_names.append("/gpu:0")
df = get_times(args.max_time, matrix_sizes, device_names)
print(df)

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
