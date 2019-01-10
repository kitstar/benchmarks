#!/bin/bash

BATCH_SIZE=8
MODEL=resnet50
NUM_GPUS=0
#NUM_WORKERS=1
UPDATE_METHOD=independent
#UPDATE_METHOD=collective_all_reduce
#UPDATE_METHOD=replicated
UPDATE_METHOD=parameter_server
#REDUCE=pscpu:32k:xring
#REDUCE=xring
#REDUCE=collective
#REDUCE=nccl

python tf_cnn_benchmarks.py \
--variable_update=${UPDATE_METHOD} \
--all_reduce_spec=${REDUCE} \
--batch_size=${BATCH_SIZE} \
--model=${MODEL} \
--num_batches=1000 \
--num_gpus=${NUM_GPUS} \
--local_parameter_device=cpu
# --use_fp16
#--num_workers=${NUM_WORKERS} \

