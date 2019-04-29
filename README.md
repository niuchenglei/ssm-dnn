Structured Semantic Model supported Deep Neural Network for Click-Through Rate Prediction

# File Description

* ```./wide_deep.py``` : Traditional official Wide\&Deep 

* ```./wide_deep_emb_conv.py``` : Wide\&Deep with convolution and pooling (SSM main code)

* ```./sparse_array_categorical_column.py``` : Sparse array feature with CSV format input

* ```./submit_local.sh``` : Run local

# How to run
1. Run ```sh submit_local.sh```

# Sample Submit Command

```
type=wide_deep_conv   # wide, deep, wide_deep, wide_deep_conv are available
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
```

# Submit Command Tutorial

* ### --pretrain

    * Indicates pretrain model or not.

* ### --embedding_model

    * The tf model file pre-trained. 

# Model Structure
1. How SSM interect with Wide\&Deep.
<img src="images/wide_deep_ssm.png" width = "400" height = "430" div align=center />
2. How to construct convolution sequences.
<img src="images/ssm.png" width = "400" height = "226" div align=center />
3. How "Delay Convolution" works, and it performs better than traditional conv-pool-conv-pool methods cause brings feature relations(convolutions) between different scale(poolings), and more powerful and efficient.
<img src="images/basis_scale.png" width = "250" height = "333" div align=center />
4. What is the SSM output vector looks like, we use t-sne to mapping high-dimension into 2-d graph to figure out what SSM learned from those feature embeddings.
<img src="images/diagram_en.png" width = "400" height = "311" div align=center /> 
