#!/bin/bash
Python=`which python`

type=$1
if [ -z $type ]; then
    echo "type is empty $type"
    exit 100
fi

export HADOOP_HDFS_HOME=/usr/local/hadoop-2.7.3/
export LD_LIBRARY_PATH=/usr/local/hadoop-2.7.3/lib/native/:/usr/local/jdk1.8.0_131/jre/lib/amd64/server/
sample_data=/data0/chenglei3/train_full.txt
eval_data=/data0/chenglei3/test_full.txt
model_dir=model_zoo/${type}/

rm ${model_dir}/* -Rf

#$Python run_local.py \
python/bin/python3 wide_deep_emb_conv.py \
    --checkpoints_dir=$model_dir \
    --save_checkpoints_steps=1000 \
    --batch_size=10000 \
    --num_epochs=10000000 \
    --data_dir=$sample_data \
    --validate_dir=$eval_data \
    --shuffle_buffer_size=10000 \
    --embedding_model=./model_zoo/wide_deep_emb_conv_cluster/model.ckpt-114471 \
    --pretrain=no \
    --model_type=${type}

#--embedding_model=./model_zoo/wide_deep_emb_conv_cluster/model.ckpt-114471 \

#python/python/bin/python3 wide_deep_emb_conv.py --task=dump


