from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import os, sys, json
import abc
import collections
import math

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib

from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_embedding():
    tf.set_random_seed(1)
    color_data = {'color': [['R','R'],['G','G'],['B','B'],['A','A'],['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_tensor = color_column._get_sparse_tensors(builder)

    color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

test_embedding()



