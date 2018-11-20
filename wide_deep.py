# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import os, sys, json

import tensorflow as tf
#ages,age,gender,platform,phone,location,network,bidtype,psid,style,link,show,position,zerocate,fircate,seccat,hierarchy_smooth_ctr,history_ctr,gender_feed_smooth_ctr,gender_cust_smooth_ctr,platform_feed_smooth_ctr,platform_cust_smooth_ctr,cust60_smooth_ctr,custid,adid,feedid
_CSV_COLUMNS = [
    'ages','age','gender','platform','phone','location','network','bidtype','psid','style','link','show','position','zerocate','fircate','seccate','hierarchy_smooth_ctr','history_ctr','gender_feed_smooth_ctr','gender_cust_smooth_ctr','platform_feed_smooth_ctr','platform_cust_smooth_ctr','cust60_smooth_ctr','custid','adid','feedid','label'
]
_CSV_COLUMN_DEFAULTS = [[''], [1008], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [0], [0], [0], [0], [0], [0], [0], [''], [''], [''], [0]]
_GLOBAL_FEATURES = []

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
    flags.DEFINE_integer("num_epochs"            , 1000000         , "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size"            , 10000           , "Training batch size, default 512")
    flags.DEFINE_integer("shuffle_buffer_size"   , 10000           , "dataset shuffle buffer size, default 10000")
    flags.DEFINE_float("learning_rate"           , 0.01            , "Learning rate, default 0.01")
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


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    age = tf.feature_column.numeric_column('age')
    ages = tf.feature_column.categorical_column_with_vocabulary_list('ages', [
        '1101', '1102', '1103', '1104', '1105', '1106'])
    gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['400','401','402'])
    platform = tf.feature_column.categorical_column_with_vocabulary_list('platform', ['', '90110100','90110200','90110201','90110202'])
    phone = tf.feature_column.categorical_column_with_vocabulary_list('phone', ['','90101000','90102000','90103000','90104000','90105000','90106000','90107000','90108000','90109000','90110000','90199000'])
    location = tf.feature_column.categorical_column_with_vocabulary_list('location', ['','30000','30101','30102','30103','30104','30105','30106','30107','30108','30109','30110','30111','30112','30113','30114','30115','30116','30117','30118','30119','30120','30121','30201','30301','30401','30402','30403','30404','30405','30406','30407','30408','30409','30410','30411','30501','30502','30503','30504','30505','30506','30507','30508','30509','30510','30511','30512','30513','30601','30602','30603','30604','30605','30606','30607','30608','30609','30610','30611','30612','30613','30614','30615','30616','30617','30701','30702','30703','30704','30705','30706','30707','30708','30709','30801','30802','30803','30804','30805','30806','30807','30808','30809','30810','30811','30812','30813','30814','30815','30816','30817','30818','30819','30820','30821','30900','30901','30902','30903','30904','30905','30906','30907','30908','30909','30910','30911','30912','30913','30914','31000','31001','31002','31003','31004','31005','31006','31007','31008','31009','31010','31011','31012','31013','31014','31015','31016','31017','31101','31102','31103','31104','31105','31106','31107','31108','31109','31110','31111','31112','31113','31114','31201','31202','31203','31204','31205','31206','31207','31208','31209','31210','31211','31301','31302','31303','31304','31305','31306','31307','31308','31309','31310','31311','31312','31313','31314','31401','31402','31403','31404','31405','31406','31407','31408','31409','31410','31501','31502','31503','31504','31505','31506','31507','31508','31509','31510','31511','31512','31513','31514','31601','31701','31702','31703','31704','31705','31706','31707','31708','31709','31710','31711','31712','31713','31714','31715','31716','31717','31801','31901','31902','31903','31904','31905','31906','31907','31908','31909','31910','31911','31912','31913','32001','32002','32003','32004','32005','32006','32007','32008','32009','32010','32011','32100','32101','32102','32103','32104','32105','32106','32107','32108','32109','32200','32201','32202','32203','32204','32205','32206','32207','32208','32209','32210','32211','32212','32213','32214','32215','32216','32301','32302','32303','32304','32305','32306','32307','32308','32309','32310','32311','32401','32402','32403','32404','32405','32406','32407','32408','32409','32501','32600','32601','32602','32603','32604','32605','32606','32607','32608','32609','32610','32611','32612','32701','32800','32801','32802','32900','32901','32902','32903','32904','32905','32906','32907','32908','32909','32910','32911','32912','32913','32914','32915','32916','32917','33001','33002','33003','33004','33005','33006','33007','33008','33009','33010','33011','33012','33013','33014','33101','33102','33103','33104','33105','33201','33300','33301','33302','33303','33304','33305','33306','33307','33308','33401','33402','33403','33404','33405','33406','33407','33501','33601','33701','33801','33901','34001','34101','34201'])
    network = tf.feature_column.categorical_column_with_vocabulary_list('network', ['','60110100','60110200','60110302','60110303','60110304'])
    bidtype = tf.feature_column.categorical_column_with_vocabulary_list('bidtype', ['','60211','60214','60218','602112'])
    psid = tf.feature_column.categorical_column_with_vocabulary_list('psid', ['','70101011','70101012','70101031','70102023','70104011','70106011','70110011'])

    style = tf.feature_column.categorical_column_with_vocabulary_list('style', ['','702100','702101','702102','702104','702105','702106','702110','702112','702113','702114','702115','702119','702120','702122','702124','702130','702140','702150','702152','702170','702197','702198'])
    link = tf.feature_column.categorical_column_with_vocabulary_list('link', ['','80210','80211','80212','80213','80214','80215','80216','80217','80218'])
    show = tf.feature_column.categorical_column_with_vocabulary_list('show', ['','90210','90211'])

    position = tf.feature_column.categorical_column_with_vocabulary_list('position', ['','7100212','7100213','7100214','7100215','7100216','7100217','7100218','7100219','71002110','71002111','71002112','71002113','71002114','71002115','71002116','71002117','71002118','71002119','71002120','71002121','71002122','71002123','71002124','71002125','71002126','71002127','71002128','71002129','71002130','71002131','71002132','71002133','71002134','71002135','71002136','71002137','71002138','71002139','71002140','71002141','71002142','71002143','71002144','71002145','71002146','71002147','71002148','71002149','71002150','71002151','71002152','71002153','71002155','71002157','71002158','71002159','71002163','71002166','71002167','71002168','71002173','71002175','71002181','71002183','71002191','71002193'])

    zerocate = tf.feature_column.categorical_column_with_vocabulary_list('zerocate', ['','null','20000000','29000000','31000000','32000000','33000000','34000000','35000000'])
    fircate = tf.feature_column.categorical_column_with_vocabulary_list('fircate', ['','0','200100','200200','200300','200500','200600','200700','200800','200900','201000','201400','201600','201700','201800','202200','299900','31001000','31002000','32001000','33001000','33002000','33003000','33004000','33005000','33006000','33007000','33008000','33009000','34001000','34002000','34003000','34004000','34005000','34006000','34007000','34008000','34010000','34011000','34012000','35001000','35003000','35004000','35005000','35006000'])
    seccate = tf.feature_column.categorical_column_with_vocabulary_list('seccate', ['','0','200101','200103','200199','200201','200202','200203','200204','200205','200206','200207','200208','200209','200210','200299','200399','200599','200601','200602','200604','200605','200607','200608','200609','200701','200802','200803','200804','200805','200899','200906','200907','201006','201401','201403','201404','201405','201499','201601','201604','201605','201699','201701','201702','201705','201799','201808','201899','202299','299999','31001001','31001002','31001003','31001004','31001006','31001007','31001008','31001010','31001011','31001012','31001014','31001015','31001016','31001017','31001018','31001019','31001020','31001021','31001023','31001024','31001026','31002001','31002002','31002004','31002008','31002009','31002010','31002011','31002014','31002015','32001001','32001002','32001003','32001004','32001005','32001006','32001009','32001010','32001011','32001014','32001015','33001001','33001002','33001003','33001004','33001005','33001006','33001009','33001013','33001014','33001015','33002001','33002002','33002003','33002004','33002005','33002006','33002008','33002009','33002010','33002011','33002012','33002013','33003001','33003002','33003003','33003004','33003005','33004001','33004002','33004003','33004004','33004006','33004007','33004008','33004009','33005001','33005002','33005003','33005004','33005005','33006002','33006003','33006004','33006005','33006007','33006008','33006009','33007001','33007005','33007007','33008001','33008002','33008003','33008004','33009001','33009002','34001001','34001003','34001004','34002001','34002002','34002003','34002004','34002005','34002006','34002007','34002008','34002009','34002010','34002012','34003001','34004001','34004002','34004003','34004004','34004005','34004006','34005001','34005002','34005003','34006001','34006002','34006003','34006004','34006005','34007001','34008001','34008003','34008004','34010001','34010004','34010005','34010006','34010007','34010008','34010009','34010010','34010012','34010014','34010015','34010016','34011001','34012001','34012002','34012003','34012004','34012006','34012007','34012008','35001001','35003001','35003005','35003007','35004001','35005004','35005005','35005006','35005007','35005008','35005009','35005010','35006001','35006002','35006003','35006004','35006005','35006006','35006007','35006008','35006009','35006010','35006011','35006012','35006013'])

    hierarchy_smooth_ctr = tf.feature_column.numeric_column('hierarchy_smooth_ctr')
    history_ctr = tf.feature_column.numeric_column('history_ctr')
    gender_feed_smooth_ctr = tf.feature_column.numeric_column('gender_feed_smooth_ctr')
    gender_cust_smooth_ctr = tf.feature_column.numeric_column('gender_cust_smooth_ctr')
    platform_feed_smooth_ctr = tf.feature_column.numeric_column('platform_feed_smooth_ctr')
    platform_cust_smooth_ctr = tf.feature_column.numeric_column('platform_cust_smooth_ctr')
    cust60_smooth_ctr = tf.feature_column.numeric_column('cust60_smooth_ctr')

    custid = tf.feature_column.categorical_column_with_hash_bucket('custid', hash_bucket_size=100000)
    adid = tf.feature_column.categorical_column_with_hash_bucket('adid', hash_bucket_size=1000000)
    feedid = tf.feature_column.categorical_column_with_hash_bucket('feedid', hash_bucket_size=1000000)

    '''
    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    '''

    # Wide columns and deep columns.
    base_columns = [
        age, ages, gender, platform, phone, location, network, bidtype, psid, style, link, show, position, zerocate, fircate, seccate, custid, adid, feedid, 
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['network', 'psid'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['style', 'zerocate'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['style', 'fircate'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['style', 'ages'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['style', 'gender'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['style', 'psid'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['ages', 'platform', 'psid'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    # age, ages, gender, platform, phone, location, network, bidtype, psid, style, link, show, position, zerocate, fircate, seccate, custid, adid, feedid,
    deep_columns = [
        age,
        hierarchy_smooth_ctr,
        history_ctr,
        gender_feed_smooth_ctr,
        gender_cust_smooth_ctr,
        platform_feed_smooth_ctr,
        platform_cust_smooth_ctr,
        cust60_smooth_ctr,
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(phone),
        tf.feature_column.indicator_column(bidtype),
        tf.feature_column.indicator_column(position),
        tf.feature_column.indicator_column(style),
        tf.feature_column.indicator_column(link),
        tf.feature_column.indicator_column(show),
        tf.feature_column.indicator_column(ages),
        # To show an example of embedding
        tf.feature_column.embedding_column(seccate, dimension=8),
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    set_tfconfig_environ()

    wide_columns, deep_columns = build_model_columns()
    global _GLOBAL_FEATURES
    _GLOBAL_FEATURES = wide_columns + deep_columns
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_secs = FLAGS.save_checkpoints_steps, #300
        keep_checkpoint_max = 3,
        model_dir=model_dir)

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)

def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    return features, tf.equal(labels, 1)

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
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    results = model.evaluate(input_fn=input_fn_for_eval)
    for key in sorted(results): print('%s: %s' % (key, results[key]))
    print("after evaluate")

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


if __name__ == '__main__':
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
