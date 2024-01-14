#!/bin/bash
GPU=${1:-"0"}
TASK=${2:-"mnli"}
BATCH=${3:-"256"}

if [ "$TASK" = "mnli" ]; then
  TASK_DIR="MNLI"
  LENGTH_CONFIG="49,28,23,18,10,1"
  AVG_TIME=5
elif [ "$TASK" = "qnli" ]; then
  TASK_DIR="QNLI"
  LENGTH_CONFIG="56,32,21,18,14,1"
  AVG_TIME=5
elif [ "$TASK" = "qqp" ]; then
  TASK_DIR="QQP"
  LENGTH_CONFIG="38,22,15,10,5,1"
  AVG_TIME=3
elif [ "$TASK" = "sst2" ]; then
  TASK_DIR="SST-2"
  LENGTH_CONFIG="26,12,12,8,3,1"
  AVG_TIME=10
elif [ "$TASK" = "cola" ]; then
  TASK_DIR="CoLA"
  LENGTH_CONFIG="21,15,13,10,5,1"
  AVG_TIME=10
elif [ "$TASK" = "mrpc" ]; then
  TASK_DIR="MRPC"
  LENGTH_CONFIG="53,30,16,10,5,1"
  AVG_TIME=10
elif [ "$TASK" = "rte" ]; then
  TASK_DIR="RTE"
  LENGTH_CONFIG="42,30,20,11,5,1"
  AVG_TIME=10
elif [ "$TASK" = "stsb" ]; then
  TASK_DIR="STS-B"
  LENGTH_CONFIG="45,29,18,14,7,1"
  AVG_TIME=10
fi

echo "RUNNING ${TASK} on GPU ${GPU}"
MODEL_DIR=path_to_model
OUTPUT_DIR=path_to_output
CUDA_VISIBLE_DEVICES=${GPU} python run_glue.py \
  --do_eval \
  --do_predict \
  --model_name_or_path ${MODEL_DIR} \
  --task_name ${TASK} \
  --data_dir ../glue_data/${TASK_DIR} \
  --max_seq_length 128 \
  --length_adaptive \
  --output_dir ${OUTPUT_DIR} \
  --per_device_eval_batch_size ${BATCH} \
  --avg_time ${AVG_TIME} \
  --eval_length_config ${LENGTH_CONFIG} \
