#!/bin/bash

BATCH_SIZE=128
MODEL=resnet50
NUM_GPUS=1

python tf_cnn_benchmarks.py \
--variable_update=independent \
--data_format=NHWC \
--batch_size=${BATCH_SIZE} \
--model=${MODEL} \
--num_gpus=${NUM_GPUS} \
--num_batches=500 \
--forward_only=True \
--local_parameter_device=cpu \
--use_fp16=False
