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

class _SparseArrayCategoricalColumn(fc_core._CategoricalColumn, collections.namedtuple('_SparseArrayCategoricalColumn', ['key', 'num_buckets', 'category_delimiter'])):

    @property
    def name(self):
        return self.key

    @property
    def _parse_example_spec(self):
        return {self.key: parsing_ops.VarLenFeature(dtypes.string)}

    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        flat_input = array_ops.reshape(input_tensor, (-1,))
        input_tensor = tf.string_split(flat_input, self.category_delimiter)

        if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError('SparseColumn input must be a SparseTensor.')

        sparse_values = input_tensor.values
        # tf.summary.text(self.key, flat_input)
        sparse_id_values = string_ops.string_to_hash_bucket_fast(sparse_values, self.num_buckets, name='lookup')
        return sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

    @property
    def _variable_shape(self):
        if not hasattr(self, '_shape'):
            self._shape = tensor_shape.vector(self.num_buckets)
        return self._shape

    @property
    def _num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.num_buckets

    def _get_sparse_tensors(self, inputs, weight_collections=None,
                            trainable=None):
        return fc_core._CategoricalColumn.IdWeightPair(inputs.get(self), None)

def categorical_column_with_array_input(key, num_buckets, category_delimiter="|"):
    if (num_buckets is None) or (num_buckets < 1):
        raise ValueError('Invalid num_buckets {}.'.format(num_buckets))
    return _SparseArrayCategoricalColumn(key, num_buckets, category_delimiter)


