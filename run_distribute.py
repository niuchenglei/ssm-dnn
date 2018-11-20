# -*- coding: utf-8 -*-
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import tensorflow as tf
from tensorflow import feature_column as fc
import sys
#import utility
#import data_utility


def define_flags():
    flags = tf.app.flags
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
    flags.DEFINE_integer("train_steps"           , 1000000         , "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size"            , 512             , "Training batch size, default 512")
    flags.DEFINE_integer("shuffle_buffer_size"   , 10000           , "dataset shuffle buffer size, default 10000")
    flags.DEFINE_float("learning_rate"           , 0.01            , "Learning rate, default 0.01")
    flags.DEFINE_float("dropout_rate"            , 0.25            , "Drop out rate, default 0.25")
    flags.DEFINE_integer("num_parallel_readers"  , 5               , "number of parallel readers for training data, default 5")
    flags.DEFINE_integer("save_checkpoints_steps", 5000            , "Save checkpoints every this many steps, default 5000")
    flags.DEFINE_boolean("run_on_cluster"        , False           , "Whether the cluster info need to be passed in as input, default False")
    flags.DEFINE_string("input_strategy"         , "local"         , "Coorpate with \"input-strategy\", input data strategy")
    
    FLAGS = flags.FLAGS
    return FLAGS

FLAGS = define_flags()
my_feature_columns = []


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
  
    #if "INPUT_FILE_LIST" in os.environ:
    #    INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    #    if INPUT_PATH:
    #        print("input path:", INPUT_PATH)
    #        FLAGS.data_dir = INPUT_PATH.get(FLAGS.data_dir)
    #        FLAGS.validate_dir = INPUT_PATH.get(FLAGS.validate_dir)
    #    else:  # for ps
    #        print("load input path failed.")
    #        FLAGS.train_data = None
    #        FLAGS.eval_data = None


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


def create_feature_columns_old():
    feed_id = fc.embedding_column(fc.categorical_column_with_hash_bucket("feed_id", 10240), 64, combiner='sum')
    cust_uid = fc.embedding_column(fc.categorical_column_with_hash_bucket("cust_uid", 10240), 64, combiner='sum')#, dtype=tf.int64), 64, combiner='sum')
    adid = fc.embedding_column(fc.categorical_column_with_hash_bucket("adid", 10240), 64, combiner='sum')#, dtype=tf.int64), 64, combiner='sum')
    network_type = fc.indicator_column(fc.categorical_column_with_vocabulary_list("network_type", 
        vocabulary_list=(60110100, 60110200, 60110301, 60110302, 60110303, 60110304, 60110305), dtype=tf.int64, default_value=0))
    psid = fc.indicator_column(fc.categorical_column_with_vocabulary_list("psid", 
        vocabulary_list=(70101011, 70101012, 70101013, 70101021, 70101031, 70101032, 70101033, 70102011, 70102021, 70102022,
         70102023, 70102024, 70102025, 70102026, 70102027, 70103011, 70103012, 70103021, 70103031, 70104011,
         70104012, 70104021, 70105011, 70105021, 70105022, 70106011, 70107011, 70109011, 70109021, 70110011), dtype=tf.int64, default_value=0)) 
    uid = fc.embedding_column(fc.categorical_column_with_hash_bucket("uid", 10240), 64, combiner='sum')#, dtype=tf.int64), 64, combiner='sum')
    age = fc.indicator_column(fc.categorical_column_with_hash_bucket("age", 100, dtype=tf.int64))
    gender = fc.indicator_column(fc.categorical_column_with_vocabulary_list("gender", 
        vocabulary_list=(400, 401, 402), dtype=tf.int64, default_value=0))
    promotion_objective = fc.indicator_column(fc.categorical_column_with_vocabulary_list("promotion_objective", 
        vocabulary_list=(88010001, 88010002, 88010003, 88010004, 88020001, 88020002, 88020003, 88020004, 88020005, 88020006), dtype=tf.int64, default_value=0))
    bid_type = fc.indicator_column(fc.categorical_column_with_vocabulary_list("bid_type", 
        vocabulary_list=(1, 2, 4, 8, 12), dtype=tf.int64, default_value=0))
    creative_style = fc.indicator_column(fc.categorical_column_with_hash_bucket("creative_style", 100, dtype=tf.int64))
    location = fc.indicator_column(fc.categorical_column_with_hash_bucket("location", 700, dtype=tf.int64))
    appid = fc.embedding_column(fc.categorical_column_with_hash_bucket("appid", 10240), 64, combiner='sum')#, dtype=tf.int64), 64, combiner='sum')
    platform = fc.indicator_column(fc.categorical_column_with_vocabulary_list("platform", 
        vocabulary_list=(90000000, 90110100, 90110200, 90110201, 90110202), dtype=tf.int64, default_value=0))
    brand = fc.indicator_column(fc.categorical_column_with_vocabulary_list("brand", 
        vocabulary_list=(90101000, 90102000, 90103000, 90104000, 90105000, 90106000, 90107000, 90108000, 90109000, 90110000, 90199000), dtype=tf.int64, default_value=0))

    global my_feature_columns

    my_feature_columns = [network_type, psid, age, gender, promotion_objective, bid_type, creative_style, location, platform, brand]
    print("feature columns:", my_feature_columns)
    return my_feature_columns


def create_feature_columns():
    feed_id = fc.categorical_column_with_hash_bucket("feed_id", 10240)
    feed_id_embed = fc.embedding_column(feed_id, 64, combiner='sum')

    cust_uid = fc.categorical_column_with_hash_bucket("cust_uid", 10240)
    cust_uid_embed = fc.embedding_column(cust_uid, 64, combiner='sum')

    adid = fc.categorical_column_with_hash_bucket("adid", 10240)
    adid_embed = fc.embedding_column(adid, 64, combiner='sum')

    network_type = fc.categorical_column_with_vocabulary_list("network_type", 
        vocabulary_list=('60110100', '60110200', '60110301', '60110302', '60110303', '60110304', '60110305'), default_value=0)
    network_type_ind = fc.indicator_column(network_type)

    psid = fc.categorical_column_with_vocabulary_list("psid", 
        vocabulary_list=('70101011', '70101012', '70101013', '70101021', '70101031', '70101032', '70101033', '70102011', '70102021', '70102022',
         '70102023', '70102024', '70102025', '70102026', '70102027', '70103011', '70103012', '70103021', '70103031', '70104011',
         '70104012', '70104021', '70105011', '70105021', '70105022', '70106011', '70107011', '70109011', '70109021', '70110011'), default_value=0)
    psid_ind = fc.indicator_column(psid) 

    uid = fc.categorical_column_with_hash_bucket("uid", 10240)
    uid_embed = fc.embedding_column(uid, 64, combiner='sum')

    age = fc.categorical_column_with_hash_bucket("age", 100)
    age_ind = fc.indicator_column(age)

    gender = fc.categorical_column_with_vocabulary_list("gender", 
        vocabulary_list=('400', '401', '402'), default_value=0)
    gender_ind = fc.indicator_column(gender)

    promotion_objective = fc.categorical_column_with_vocabulary_list("promotion_objective", 
        vocabulary_list=('88010001', '88010002', '88010003', '88010004', '88020001', '88020002', 
            '88020003', '88020004', '88020005', '88020006'), default_value=0)
    promotion_objective_ind = fc.indicator_column(promotion_objective)

    bid_type = fc.categorical_column_with_vocabulary_list("bid_type", 
        vocabulary_list=('1', '2', '4', '8', '12'), default_value=0)
    bid_type_ind = fc.indicator_column(bid_type)

    creative_style = fc.categorical_column_with_hash_bucket("creative_style", 100)
    creative_style_ind = fc.indicator_column(creative_style)

    location = fc.categorical_column_with_hash_bucket("location", 700)
    location_ind = fc.indicator_column(location)

    appid = fc.categorical_column_with_hash_bucket("appid", 10240)
    appid = fc.embedding_column(appid, 64, combiner='sum')

    platform = fc.categorical_column_with_vocabulary_list("platform", 
        vocabulary_list=('90000000', '90110100', '90110200', '90110201', '90110202'), default_value=0)
    platform_ind = fc.indicator_column(platform)

    brand = fc.categorical_column_with_vocabulary_list("brand", 
        vocabulary_list=('90101000', '90102000', '90103000', '90104000', '90105000', 
            '90106000', '90107000', '90108000', '90109000', '90110000', '90199000'), default_value=0)
    brand_ind = fc.indicator_column(brand)

    # cross features
    platform_appid = fc.indicator_column(fc.crossed_column(['platform', 'appid'], hash_bucket_size=10240))
    psid_cust_uid = fc.indicator_column(fc.crossed_column(['psid', 'cust_uid'], hash_bucket_size=10240))
    creative_style_psid = fc.indicator_column(fc.crossed_column(['creative_style', 'psid'], hash_bucket_size=10240))
    creative_style_gender = fc.indicator_column(fc.crossed_column(['creative_style', 'gender'], hash_bucket_size=10240))
    creative_style_brand = fc.indicator_column(fc.crossed_column(['creative_style', 'brand'], hash_bucket_size=10240))
    creative_style_location = fc.indicator_column(fc.crossed_column(['creative_style', 'location'], hash_bucket_size=10240))
    creative_style_network_type = fc.indicator_column(fc.crossed_column(['creative_style', 'network_type'], hash_bucket_size=10240))
    platform_appid_gender = fc.indicator_column(fc.crossed_column(['platform', 'appid', 'gender'], hash_bucket_size=10240))
    platform_appid_location = fc.indicator_column(fc.crossed_column(['platform', 'appid', 'location'], hash_bucket_size=10240))
    platform_appid_brand = fc.indicator_column(fc.crossed_column(['platform', 'appid', 'brand'], hash_bucket_size=10240))
    platform_appid_creative_style = fc.indicator_column(fc.crossed_column(['platform', 'appid', 'creative_style'], hash_bucket_size=10240))
    cross_f_2 = [platform_appid, psid_cust_uid, creative_style_psid, creative_style_gender, 
                 creative_style_brand, creative_style_location, creative_style_network_type]
    cross_f_3 = [platform_appid_gender, platform_appid_location, platform_appid_brand, platform_appid_creative_style]

    global my_feature_columns

    my_feature_columns = [feed_id_embed, cust_uid_embed, adid_embed, network_type_ind, psid_ind, uid_embed, 
                age_ind, gender_ind, promotion_objective_ind, bid_type_ind, 
                creative_style_ind, location_ind, platform_ind, brand_ind]
    my_feature_columns += cross_f_2
    my_feature_columns += cross_f_3
    print("feature columns:", my_feature_columns)
    return my_feature_columns



def parse_exmp(serial_exmp):
    click = fc.numeric_column("clk", default_value=0, dtype=tf.int64)
    pay = fc.numeric_column("cov", default_value=0, dtype=tf.int64)
    fea_columns = [click, pay]
    fea_columns += my_feature_columns
    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    feats = tf.parse_single_example(serial_exmp, features=feature_spec)
    click = feats.pop('clk')
    pay = feats.pop('cov')
    return feats, {'ctr': tf.to_float(click), 'cvr': tf.to_float(pay)}


def train_input_fn(filenames, batch_size, shuffle_buffer_size):
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)
    dataset = dataset.repeat(5).batch(batch_size).prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def eval_input_fn(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)#8
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(batch_size)
    # Return the read end of the pipeline.
    return dataset


def build_mode(features, mode, params):
    net = fc.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
      if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Compute logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def my_model(features, labels, mode, params):
    with tf.variable_scope('ctr_model'):
        ctr_logits = build_mode(features, mode, params)
    with tf.variable_scope('cvr_model'):
        cvr_logits = build_mode(features, mode, params)

    # define ctr, cvr, ctcvr
    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
          'probabilities': prop,
          'ctr_probabilities': ctr_predictions,
          'cvr_probabilities': cvr_predictions
        }
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # define loss
    y1 = labels['cvr']
    y2 = labels['ctr']
    print(str(prop)+'\n\n'+str(y1))
    # p-ctcvr and cvr
    cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y1, prop), name="cvr_loss")
    # p-ctr and ctr
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y2, logits=ctr_logits), name="ctr_loss")
    loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")
    
    ctr_accuracy = tf.metrics.accuracy(labels=y2, predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(labels=y1, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
    ctr_auc = tf.metrics.auc(y2, ctr_predictions)
    cvr_auc = tf.metrics.auc(y1, prop)
    metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "ctr_auc" : ctr_auc[1], "cvr_auc": cvr_auc[1]}, every_n_iter=20)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, training_hooks = [logging_hook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    #optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    #optimizer = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics, training_hooks = [logging_hook])



def main(unused_argv):
    set_tfconfig_environ()
    create_feature_columns()
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        #config=tf.estimator.RunConfig(model_dir=FLAGS.train_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
        config=tf.estimator.RunConfig(model_dir=FLAGS.checkpoints_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )
    batch_size = FLAGS.batch_size
    #print("model_dir is :%s" % os.path.abspath(FLAGS.train_dir))
    print("train steps: %s, batch_size:%s" % (FLAGS.train_steps, batch_size))
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
    shuffle_buffer_size = FLAGS.shuffle_buffer_size
    print("data_dir: %s" % train_files)
    print("validate_dir:%s" % eval_files)
    print("shuffle_buffer_size%s" % shuffle_buffer_size)

    # train process
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_files, batch_size, shuffle_buffer_size),
        max_steps=FLAGS.train_steps
    )
    input_fn_for_eval = lambda: eval_input_fn(eval_files, batch_size)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    results = classifier.evaluate(input_fn=input_fn_for_eval)
    for key in sorted(results): print('%s: %s' % (key, results[key]))
    print("after evaluate")
    
    if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        print("exporting model ...")
        feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
        print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    print("quit main")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

