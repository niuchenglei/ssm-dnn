from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import os, sys, json, itertools
import numpy as np
import pickle

import tensorflow as tf
from sparse_array_categorical_column import categorical_column_with_array_input

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

_CSV_COLUMNS = ['ages','age','gender','platform','phone','location','network','bidtype','psid','style','link','show','position','zerocate','fircate','seccate','hierarchy_smooth_ctr','history_ctr','gender_feed_smooth_ctr','gender_cust_smooth_ctr','platform_feed_smooth_ctr','platform_cust_smooth_ctr','cust60_smooth_ctr','custid','adid','feedid','user_class','cust_tag','feed_word','user_bhvtag','user_bhvword','label']
_CSV_COLUMN_DEFAULTS = [[''], [1008], [''],[''],[''],[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [0], [0], [0], [0], [0], [0], [0], ['0'], ['0'], ['0'], ['0'],['0'], ['0'], ['0'],['0'], [0]]

def define_flags():
    flags = tf.app.flags
    tf.app.flags.DEFINE_string("task"        , "train", "train/dump/pred")
    # Flags Sina ML required
    tf.app.flags.DEFINE_string("data_dir"    , "", "Set local data path of train set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("validate_dir", "", "Set local data path of validate set. Coorpate with 'input-strategy DOWNLOAD'.")
    flags.DEFINE_integer("num_epochs"            , 100         , "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size"        , 1000   , '')

    FLAGS = flags.FLAGS
    return FLAGS

FLAGS = define_flags()

def freeze_graph(output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if (output_node_names is None):
        output_node_names = 'loss'

    if not tf.gfile.Exists(my_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % my_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(my_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_frozen_graph(prefix="frozen_graph"):
    frozen_graph_filename = os.path.join(my_dir, "frozen_model.pb")

    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name=prefix)

    return graph

# *****************************************************************************

if __name__ == '__main__':
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    tf.logging.set_verbosity(tf.logging.INFO)

    my_dir = 'model_zoo/wide_deep_emb_conv_cluster/'
    prefix = 'frozen_graph'
    freeze_graph("pred,emb_conv_model/flatten_layer")

    graph = load_frozen_graph(prefix=prefix)

    for op in graph.get_operations():
        print(op.name)

    #exit(0)
    with open(os.path.join(my_dir, 'input_tensor_map.pickle'), 'rb') as f:
        input_tensor_map = pickle.load(f)

    batch_size = FLAGS.batch_size
    train_fp = open('/data4/ads_fst/chenglei3/validate_0.txt', 'r')
    train_out_fp = open('/data4/ads_fst/chenglei3/validate_0.bin', 'wb')
    '''
    f = file("tmp.bin","wb")
    np.save(f,a)
    np.save(f,b)
    np.save(f,c)
    f.close()
    f = file("tmp.bin","rb")
    aa = np.load(f)
    bb = np.load(f)
    cc = np.load(f)
    f.close()
    '''
    X_test = {'user_class':['读书阅读|时尚美妆|商场打折|在线购物|家居设计|段子|广告设计|华语80后偶像|服装饰品|团购海淘'], 'cust_tag':['淘宝优惠|优惠劵|优惠券|打折|优惠|购物|优惠卷|省钱|折扣'], 'feed_word':['口红|体验|大牌'], 'user_bhvword':['diss|lamer|三观|下单|中国|人生|信号|关心|兴趣|分配|包邮|化妆|原因|双方|反应|同事|同款|名额|售价|回家|围观|坚果|垃圾|垃圾桶|大牌|大胆|姐姐|官网|宝贝|室友|小姐姐|小心|开学|情侣|感觉|房租|护肤|报价|拍一套|整箱|早安|时尚|晚安|智能|月薪|机会|机票|活动|特惠|玩具|生活|生病|男>友|男朋友|眼睛|种草|细数|美味|自动|芳心|进店|锦鲤|限时|限量|高逼格'], 'user_bhvtag':['beauty|skin|买|买东西|买买买|人生|休闲|优惠|优惠券|优惠劵|优惠卷|便宜|养生|化妆|变美|吃|妆|婚|婚庆|婚照|婚礼|婚紗|婚纱|婚纱店|婚纱拍摄|婚纱摄影|婚纱摄影店|婚纱照|实用|>彩妆|影楼|心情|心灵鸡汤|心理学|恋爱|情感|情感两性|感情|打折|折扣|护肤|护肤&美妆|护肤化妆|护肤博主|护肤彩妆|护肤美妆|拍照|摄影|>摄影化妆|摄影师|文字|新娘造型|新闻|旅拍|时尚|时尚美妆|本地生活|治愈系|淘宝|淘宝优惠|潮流|爱情|玩|生活|生活百科|白菜价|皮肤|省钱|种草|结婚|结婚照|网络红人|网购|美|美妆|美妆博主|美妆护肤|美妆时尚|美妆穿搭|美妆达人|美妆💄|美容美容护肤|美美哒|美肤|衣服|购物|购物优惠|资讯|随意|鸡汤']}

    sess = tf.Session(graph=graph)
    end_of_file = False

    u_class = ['美容|美妆|服装|饰品|服装饰品|时尚美妆', '婚恋家庭|旅游酒店|婚纱摄影|母婴育儿|婚恋交友', '新闻时事|体育运动|互联网|音乐|游戏动漫']
    c_tag = ['穿搭|时尚|美容|衣服|优惠|打折|网络红人|美容服饰|时装|fashion|搭配|美丽', '摄影|婚纱摄影|婚纱|婚礼|结婚|婚纱照|婚庆|摄影师|影楼', '摄影|婚纱摄影|婚纱|婚礼|结婚|婚纱照|婚庆|摄影师|影楼']

    sum_pos = 0
    sum_pred = 0
    cnt = 0
    empty_X = {}
    ind1 = _CSV_COLUMNS.index('user_class')
    ind2 = _CSV_COLUMNS.index('cust_tag')
    for k in _CSV_COLUMNS:
        empty_X[k] = []
    for k in range(0, FLAGS.num_epochs):
        X_validate = empty_X #{'user_class':[], 'cust_tag':[], 'feed_word':[], 'user_bhvword':[], 'user_bhvtag':[], 'label':[]}
        read_line_num = 0
        
        while True:
            line = train_fp.readline()
            if line is None:
                end_of_file = True
                break
            arr = line.split(',')
            if len(arr) != len(_CSV_COLUMNS):
                break

            match_tag = 0
            if len(u_class) != 0:
                match_cnt = 0
                match_id = -1
                for k in range(0, len(u_class)):
                    class_ = re.match(u_class[k], arr[ind1])
                    tag_ = re.match(u_tag[k], arr[ind2])
                    match_cnt += 1 if class_ is not None and tag_ is not None else 0
                    match_id = k+1 if class_ is not None and tag_ is not None else 0
                if match_cnt == 0:
                    continue
                match_tag = match_id
           
            X_validate['tag'] = match_tag
            for i in range(0, len(arr)):
                X_validate[_CSV_COLUMNS[i]].append(arr[i])

            read_line_num += 1
            if read_line_num == batch_size:
                break
        if end_of_file:
            break
        if len(X_validate['label']) < 1:
            break
        
        input_feed = dict()
        for key, tensor_name in input_tensor_map.items():
            #print("--------------tensor name---------"+tensor_name)
            tensor = graph.get_tensor_by_name(prefix + "/" + tensor_name)

            #print(str(key)+'\t'+str(tensor_name)+'\t'+str(tensor))
            input_feed[tensor] = X_validate[key]

        op_logits = graph.get_operation_by_name(prefix + "/pred").outputs[0]
        op_flatten = graph.get_operation_by_name(prefix + "/emb_conv_model/flatten_layer").outputs[0]

        logits, flatten = sess.run([op_logits, op_flatten], feed_dict=input_feed)

        [rows, cols] = flatten.shape
        label = np.array([float(x) for x in X_validate['label']]).reshape((len(X_validate['label']), 1))
        s = np.concatenate((label, logits, flatten), axis=1)

        real_ctr = np.average(s[:,0])
        pred_ctr = np.average(s[:,1])
        sum_pos += np.sum(s[:,0])
        sum_pred += np.sum(s[:,1])
        cnt += rows

        pos = s[np.where(s[:,0]==1)]
        [rows_pos, cols_pos] = pos.shape
        neg = s[np.where(s[:,0]==0)]
        neg2 = np.random.permutation(neg)
        s2 = np.vstack((pos, neg2[0:rows_pos+1,:]))
        s2 = np.random.permutation(s2)

        np.save(train_out_fp, s2)
        print('epoch: %d, shape:%s, ctr:%f, pred:%f, ctr_acc:%f, pred_acc:%f' % (k, str(np.shape(s2)), real_ctr, pred_ctr, sum_pos/cnt, sum_pred/cnt))

    train_fp.close()
    train_out_fp.close()

