#!/bin/bash
Python=`which python`

sample_data=data/1.txt
eval_data=data/2.txt
model_dir=model_zoo/conv_emb/

#$Python run_local.py \
python/python/bin/python3 conv_emb.py \
    --checkpoints_dir=$model_dir \
    --save_checkpoints_steps=1000 \
    --batch_size=1000 \
    --num_epochs=100000 \
    --data_dir=$sample_data \
    --validate_dir=$eval_data \
    --shuffle_buffer_size=100

#python/python/bin/python3 conv_emb.py --task=dump
