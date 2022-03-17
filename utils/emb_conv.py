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

_CSV_COLUMNS = [
    'v1', 'v2', 'label'
]
_CSV_COLUMN_DEFAULTS = [[''], [''], [0]]
_GLOBAL_FEATURES = []

def define_flags():
    flags = tf.app.flags
    tf.app.flags.DEFINE_string("task"        , "train", "train/dump/pred")
    # Flags Sina ML required
    tf.app.flags.DEFINE_string("data_dir"    , "", "Set local data path of train set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("validate_dir", "", "Set local data path of validate set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("train_dir"   , "", "Set model save path. Not input path")
    tf.app.flags.DEFINE_string("log_dir"     , "", "Set tensorboard even log path.")
    # Flags Sina ML required: for tf.train.ClusterSpec
    tf.app.flags.DEFINE_string("ps_hosts"    , "", "Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
    tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")
    # Flags Sina ML required:Flags for defining the tf.train.Server
    tf.app.flags.DEFINE_string("job_name"    , "", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index" , 0 , "Index of task within the job.Sina ML required arg.")

    flags.DEFINE_string("checkpoints_dir"        , "./checkpoints_dir" , "Set checkpoints path.")
    #flags.DEFINE_string("model_dir"              , "./model_dir"  , "Set checkpoints path.")
    flags.DEFINE_string("output_model"           , "./model_output", "Path to the training data.")
    flags.DEFINE_string("hidden_units"           , "512,256,128"   , "Comma-separated list of number of units in each hidden layer of the NN")
    flags.DEFINE_integer("num_epochs"            , 1000000         , "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size"            , 10000           , "Training batch size, default 512")
    flags.DEFINE_integer("shuffle_buffer_size"   , 10000           , "dataset shuffle buffer size, default 10000")
    flags.DEFINE_float("learning_rate"           , 0.001           , "Learning rate, default 0.01")
    flags.DEFINE_float("dropout_rate"            , 0.25            , "Drop out rate, default 0.25")
    flags.DEFINE_integer("num_parallel_readers"  , 5               , "number of parallel readers for training data, default 5")
    flags.DEFINE_integer("save_checkpoints_steps", 5000            , "Save checkpoints every this many steps, default 5000")
    flags.DEFINE_boolean("run_on_cluster"        , False           , "Whether the cluster info need to be passed in as input, default False")
    flags.DEFINE_string("input_strategy"         , "local"         , "Coorpate with \"input-strategy\", input data strategy")
    flags.DEFINE_string("model_type"             , "wide_deep"     , "wide/deep/wide_deep")

    FLAGS = flags.FLAGS
    return FLAGS

FLAGS = define_flags()

def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)

def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if FLAGS.job_name == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

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

def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    v_user = categorical_column_with_array_input('v1', 50000, '|')
    v_ad = categorical_column_with_array_input('v2', 50000, '|')

    '''
    embedding_initializer_user = tf.contrib.framework.load_embedding_initializer(ckpt_path='model.ckpt',
        embedding_tensor_name='emb_user_in',
        new_vocab_size=50000,
        embedding_dim=200
    )
    emb_user = tf.feature_column.embedding_column(v_user,
                                                   dimension=200,
                                                   initializer=embedding_initializer_x,
                                                   trainable=False)

    '''
    emb_user = tf.feature_column.embedding_column(v_user, dimension=200) #, tensor_name_in_ckpt='emb_user')  #ckpt_to_load_from
    emb_ad = tf.feature_column.embedding_column(v_ad, dimension=200) #, tensor_name_in_ckpt='emb_ad')

    features = [emb_user, emb_ad]

    return [emb_user], [emb_ad], features  #features

def my_model(features, labels, mode, params):
    with tf.variable_scope('emb_conv_model'):
        # features:  {'v1': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=string>, 'v2': <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=string>}
        # Tensor("emb_conv_model/input_layer/concat:0", shape=(?, 200), dtype=float32)      Tensor("emb_conv_model/input_layer_1/concat:0", shape=(?, 200), dtype=float32)
        v_user = tf.feature_column.input_layer(features, params['tensor_user'])
        v_ad = tf.feature_column.input_layer(features, params['tensor_ad'])   # shape=(?,200)
        tensor = tf.feature_column.input_layer(features, params['features'])  # shape=(?,400)

        print('tensor_user:'+str(tensor))
        with tf.variable_scope('conv'):
            t2d = tf.stack([v_user, v_ad], axis=2)   # shape=(?, 200, 2)
            kernel = tf.constant([[[1.0, 1.0], [1.0, -1.0]]], name='kernel')
            #kernel = tf.Variable(tf.random_uniform([1, 2, 2], -0.5, 0.5), trainable=True, name='kernel') # [1, 2, 2] for [row, col, kernel_num] cause the shape of t2d(v_user,v_ad) is 100*2, so conv on row-dim make no means
            t1d_conv = tf.nn.conv1d(t2d, kernel, 1, 'VALID')

        with tf.variable_scope('multiply'):
            t1d_dot = tf.expand_dims(tf.multiply(v_user, v_ad), 2)
        t1d = tf.concat([t1d_conv, t1d_dot], axis=2)
        print('t3:\t'+str(t1d_conv)+'\n\n'+str(t1d_dot)+'\n\n'+str(t1d))

        pool_1 = tf.layers.flatten(tf.nn.pool(t1d, [3], 'AVG', 'VALID', strides=[1]))
        pool_2 = tf.layers.flatten(tf.nn.pool(t1d, [7], 'AVG', 'VALID', strides=[3]))
        pool_3 = tf.layers.flatten(tf.nn.pool(t1d, [13], 'AVG','VALID', strides=[6]))
        flatten = tf.concat([pool_1, pool_2, pool_3], axis=1)  #shape=(?, 590)
        #flatten = tf.Print(flatten, [flatten, 'flatten'], message='Debug message:')
        logits = tf.layers.dense(flatten, 1, activation=None)
        print('flatten: '+str(flatten)+'\n')

    ctr_pred = tf.sigmoid(logits, name="CTR")
    #ctr_pred = tf.Print(ctr_pred, [ctr_pred, 'pred'], message='Debug message:',summarize=100)
    prop = ctr_pred

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
          'ctr_probabilities': ctr_pred,
        }
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # define loss
    y = tf.reshape(labels['label'], [-1,1])
    # p-ctcvr and cvr
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, prop), name="loss")

    ctr_auc = tf.metrics.auc(y, ctr_pred)
    ctr_avg = tf.metrics.mean(y)
    pred_avg = tf.metrics.mean(ctr_pred)
    metrics = {'auc': ctr_auc, 'ctr': ctr_avg, 'pred': pred_avg }
    tf.summary.scalar('auc', ctr_auc[1])
    tf.summary.scalar('ctr_avg', ctr_avg[1])
    tf.summary.scalar('pred_avg', pred_avg[1])
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "auc" : ctr_auc[1], "ctr" : ctr_avg[1], "pred" : pred_avg[1]}, every_n_iter=20)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, training_hooks = [logging_hook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    #optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    #optimizer = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics, training_hooks = [logging_hook])

def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    set_tfconfig_environ()

    v_user, v_ad, features = build_model_columns()
    global _GLOBAL_FEATURES
    _GLOBAL_FEATURES = v_user + v_ad
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_secs = FLAGS.save_checkpoints_steps, #300
        keep_checkpoint_max = 3,
        model_dir=model_dir)

    model = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': _GLOBAL_FEATURES,
            'tensor_user': v_user,
            'tensor_ad': v_ad,
            'features': features,
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.checkpoints_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )
    return model

#def process_list_column(list_column, dtype=tf.float32):
#    sparse_strings = tf.string_split(list_column, delimiter=":")
#    return tf.Tensor(tf.string_to_number(sparse_strings.values, out_type=dtype) 

def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    #columns[0] = process_list_column(columns[0], dtype=tf.int64)
    #columns[1] = process_list_column(columns[1], dtype=tf.int64)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    #feats, {'ctr': tf.to_float(click), 'cvr': tf.to_float(pay)}
    return features, { 'label': tf.to_float(labels) }

def input_fn(data_file, num_epochs, shuffle_buffer_size, batch_size):
    #"""Generate an input function for the Estimator."""
    #assert tf.gfile.Exists(data_file), (
    #    '%s not found. Please make sure you have either run data_download.py or '
    #    'set both arguments --train_data and --test_data.' % data_file)

    files = tf.data.Dataset.list_files(data_file)
    # Extract lines from input files using the Dataset API.
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1)) 
    #dataset = tf.data.TextLineDataset(files)

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def eval_input_fn(data_file, batch_size):
    # Extract lines from input files using the Dataset API.
    files = tf.data.Dataset.list_files(data_file)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1))
    #dataset = tf.data.TextLineDataset(files)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(batch_size)
    return dataset

def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.checkpoints_dir, ignore_errors=True)
    model = build_estimator(FLAGS.checkpoints_dir, FLAGS.model_type)
    if isinstance(FLAGS.data_dir, str) and os.path.isdir(FLAGS.data_dir):
        train_files = [FLAGS.data_dir + '/' + x for x in os.listdir(FLAGS.data_dir)] if os.path.isdir(
            FLAGS.data_dir) else FLAGS.data_dir
    else:
        train_files = FLAGS.data_dir
    if isinstance(FLAGS.validate_dir, str) and os.path.isdir(FLAGS.validate_dir):
        eval_files = [FLAGS.validate_dir + '/' + x for x in os.listdir(FLAGS.validate_dir)] if os.path.isdir(
            FLAGS.validate_dir) else FLAGS.validate_dir
    else:
        eval_files = FLAGS.validate_dir

    # train process
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_files, FLAGS.num_epochs, FLAGS.shuffle_buffer_size, FLAGS.batch_size),
        max_steps=FLAGS.num_epochs
    )
    input_fn_for_eval = lambda: eval_input_fn(eval_files, FLAGS.batch_size)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    results = model.evaluate(input_fn=input_fn_for_eval)
    for key in sorted(results): print('%s: %s' % (key, results[key]))
    print("after evaluate")

    #model.predict()

    if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        print("exporting model ...")
        feature_spec = tf.feature_column.make_parse_example_spec(_GLOBAL_FEATURES)
        print("feature spec: "+str(feature_spec))
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    print("quit main")

    '''
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
    '''

def dump():
    #files = tf.data.Dataset.list_files('model2')
    #dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1))

    with tf.Session() as sess:
        data_file = './model.ckpt-1.meta'

        saver = tf.train.import_meta_graph(data_file, clear_devices=True)
        saver.restore(sess, './model.ckpt-1')

        graph = tf.get_default_graph()
        w = []
        bias = 0

        # list all trainable variables
        dim0 = 0
        for var in tf.get_default_graph().get_collection("trainable_variables"):
            vv = sess.run(var)
            print(str(var.name) + '\t\t' + str(vv.shape)) #+str(vv))
            #idx = int(re.split('_|:', var.name)[1])
            #if var.name.split('/')[0] == 'weights':    #'weights/part_0:0':
            #    w.extend(list(vv[:,0]))
            #    dim0 += vv.shape[0]
            #if var.name == 'params/bias:0':
            #    bias = vv
        print('dim: '+str(dim0)+', len(w):'+str(len(w)))


if __name__ == '__main__':
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.task == 'dump':
        dump()
    else:
        tf.app.run(main=main)

