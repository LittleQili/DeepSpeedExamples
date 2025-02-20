#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --include localhost:4,5,6,7 \
   main.py \
   --data_path /localdata_ssd/yjdiao/models/Dahoas/rm-static \
      /localdata_ssd/yjdiao/models/Dahoas/full-hh-rlhf \
      /localdata_ssd/yjdiao/models/Dahoas/synthetic-instruct-gptj-pairwise \
      /localdata_ssd/yjdiao/models/yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /localdata_ssd/yjdiao/models/opt-1.3b \
   --per_device_train_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
   
#    --per_device_eval_batch_size 8 \
