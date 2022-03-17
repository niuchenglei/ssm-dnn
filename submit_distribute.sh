#!/bin/bash
#Python=`which python`

#input_data=hdfs://ns3-backup/user/ads_dm/warehouse/liuying15/tfRecords/chenglei/20180703
#input_data=hdfs://ns3-backup/user/ads_fst/chenglei3/sfst_ctr/igl_20180918
input_data=hdfs://ns3-backup/user/ads_fst/guojing7/user_ad/train
#test_input=hdfs://ns3-backup/user/ads_dm/warehouse/liuying15/tfRecords/test/20180702
#validate_data=hdfs://ns3-backup/user/ads_fst/chenglei3/sfst_ctr/igl_val
validate_data=hdfs://ns3-backup/user/ads_fst/guojing7/user_ad/test

hdfs_dir=hdfs://ns3-backup/user/ads_fst/chenglei3/sfst_ctr/
#hdfs_dir=hdfs://ns3-backup/user/ads_fst/chenglei3/sfst_ctr/wide

function pretrain() {
    hdfs_dir=${hdfs_dir}/wide_deep_conv
    out_model_dir=${hdfs_dir}"/pretrain/"
    hdfs dfs -rmr ${out_model_dir}
    hdfs dfs -mkdir ${out_model_dir}
    hdfs dfs -chmod 777 ${out_model_dir}

    output_model_check=${out_model_dir}/checkpoint
    output_model_file=${out_model_dir}/model
    hdfs dfs -rmr ${output_model_check}/*
    hdfs dfs -mkdir ${output_model_check}
    hdfs dfs -chmod 777 ${output_model_check}
    hdfs dfs -rmr ${output_model_file}/*
    hdfs dfs -mkdir ${output_model_file}
    hdfs dfs -chmod 777 ${output_model_file}

    upload_dir=${out_model_dir}/tmp
    hdfs dfs -rmr ${upload_dir}

$ML_HOME/bin/ml-submit \
   --app-type "tensorflow" \
   --app-name "tf_wide_deep" \
   --input-strategy DOWNLOAD \
   --output-strategy UPLOAD \
   --input $input_data/#data \
   --input $validate_data/#validate \
   --output $upload_dir/#models \
   --boardHistoryDir /user_ext/ads_fst/chenglei3/tflogs \
   --files wide_deep_emb_conv.py,sparse_array_categorical_column.py \
   --launch-cmd "Python/python/bin/python3 wide_deep_emb_conv.py \
       --batch_size=10000 \
       --checkpoints_dir=./checkpoints_dir \
       --data_dir=./data \
       --validate_dir=./validate \
       --train_dir=./models \
       --log_dir=./eventLog \
       --shuffle_buffer_size=10000 \
       --run_on_cluster=true \
       --num_epochs=100000 \
       --save_checkpoints_steps=500 \
       --checkpoints_dir=$output_model_check \
       --output_model=$output_model_file \
       --pretrain=yes" \
   --cacheArchive hdfs://ns3-backup/user/ads_dm/warehouse/liuying15/tensorflow/python_3_tensorflow18_cpu.zip#Python \
   --worker-memory 30G \
   --board-enable true \
   --worker-num 20 \
   --worker-cores 2 \
   --ps-memory 30G \
   --ps-num 2 \
   --ps-cores 1
}

function train_conv() {
    echo "[train_conv] $*"
    hdfs_dir=${hdfs_dir}/$1
    pre_trained_model_dir=${hdfs_dir}"/pretrain/checkpoint"

    files=$(hdfs dfs -ls ${pre_trained_model_dir}/*.meta | awk '{print $8;}'| tr '\n' ' ')
    model="xx"
    cur=123
    for k in $files; do
        v2=${k#*model.ckpt-*}
        v3=${v2%.meta*}
        if [[ $cur -lt $v3 ]]; then
            cur=$v3
            model=$k
            #echo "$model"
        fi
    done
    pre_trained_model="model.ckpt-${cur}"
    hdfs dfs -get ${pre_trained_model_dir}/*-${cur}* /tmp/
    local_files=$(ls /tmp/*-${cur}* | awk '{v=v","$1;} END{print substr(v,2)}')
    echo "pre_trained_model=${pre_trained_model}, local_files=${local_files}"

    out_model_dir=${hdfs_dir}"/train/"
    #hdfs dfs -rmr ${out_model_dir}
    hdfs dfs -mkdir ${out_model_dir}
    hdfs dfs -chmod 777 ${out_model_dir}

    output_model_check=${out_model_dir}/checkpoint
    output_model_file=${out_model_dir}/model
    #hdfs dfs -rmr ${output_model_check}/*
    hdfs dfs -mkdir ${output_model_check}
    hdfs dfs -chmod 777 ${output_model_check}
    #hdfs dfs -rmr ${output_model_file}/*
    hdfs dfs -mkdir ${output_model_file}
    hdfs dfs -chmod 777 ${output_model_file}

    upload_dir=${out_model_dir}/tmp
    hdfs dfs -rmr ${upload_dir}

$ML_HOME/bin/ml-submit \
   --app-type "tensorflow" \
   --app-name "tf_wide_deep" \
   --input-strategy DOWNLOAD \
   --output-strategy UPLOAD \
   --input $input_data/#data \
   --input $validate_data/#validate \
   --output $upload_dir/#models \
   --boardHistoryDir /user_ext/ads_fst/chenglei3/tflogs \
   --files wide_deep_emb_conv.py,sparse_array_categorical_column.py,${local_files} \
   --launch-cmd "Python/python/bin/python3 wide_deep_emb_conv.py \
       --batch_size=10000 \
       --checkpoints_dir=./checkpoints_dir \
       --data_dir=./data \
       --validate_dir=./validate \
       --train_dir=./models \
       --log_dir=./eventLog \
       --shuffle_buffer_size=10000 \
       --run_on_cluster=true \
       --num_epochs=1000000 \
       --save_checkpoints_steps=500 \
       --checkpoints_dir=$output_model_check \
       --output_model=$output_model_file \
       --embedding_model=./${pre_trained_model} \
       --model_type=wide_deep_conv \
       --fineturn=$2" \
   --cacheArchive hdfs://ns3-backup/user/ads_dm/warehouse/liuying15/tensorflow/python_3_tensorflow18_cpu.zip#Python \
   --worker-memory 30G \
   --board-enable true \
   --worker-num 20 \
   --worker-cores 2 \
   --ps-memory 30G \
   --ps-num 2 \
   --ps-cores 1
}

function train_widedeep() {
    echo "[train_widedeep] $*"
    hdfs_dir=${hdfs_dir}/$1
    out_model_dir=${hdfs_dir}
    #hdfs dfs -rmr ${out_model_dir}
    #hdfs dfs -mkdir ${out_model_dir}
    #hdfs dfs -chmod 777 ${out_model_dir}

    output_model_check=${out_model_dir}/checkpoint
    output_model_file=${out_model_dir}/model
    #hdfs dfs -rmr ${output_model_check}/*
    #hdfs dfs -mkdir ${output_model_check}
    #hdfs dfs -chmod 777 ${output_model_check}
    #hdfs dfs -rmr ${output_model_file}/*
    #hdfs dfs -mkdir ${output_model_file}
    #hdfs dfs -chmod 777 ${output_model_file}

    upload_dir=${out_model_dir}/tmp
    hdfs dfs -rmr ${upload_dir}

$ML_HOME/bin/ml-submit \
   --app-type "tensorflow" \
   --app-name "tf_wide_deep" \
   --input-strategy DOWNLOAD \
   --output-strategy UPLOAD \
   --input $input_data/#data \
   --input $validate_data/#validate \
   --output $upload_dir/#models \
   --boardHistoryDir /user_ext/ads_fst/chenglei3/tflogs \
   --files wide_deep_emb_conv.py,sparse_array_categorical_column.py \
   --launch-cmd "Python/python/bin/python3 wide_deep_emb_conv.py \
       --batch_size=10000 \
       --checkpoints_dir=./checkpoints_dir \
       --data_dir=./data \
       --validate_dir=./validate \
       --train_dir=./models \
       --log_dir=./eventLog \
       --shuffle_buffer_size=10000 \
       --run_on_cluster=true \
       --num_epochs=600000 \
       --save_checkpoints_steps=500 \
       --checkpoints_dir=$output_model_check \
       --output_model=$output_model_file \
       --model_type=$1" \
   --cacheArchive hdfs://ns3-backup/user/ads_dm/warehouse/liuying15/tensorflow/python_3_tensorflow18_cpu.zip#Python \
   --worker-memory 30G \
   --board-enable true \
   --worker-num 20 \
   --worker-cores 2 \
   --ps-memory 30G \
   --ps-num 2 \
   --ps-cores 1
}

#pretrain
#train_conv wide_deep_conv false
train_conv wide_deep_conv_fineturn true
#train_widedeep wide_deep



