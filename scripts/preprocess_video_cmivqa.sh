#!/bin/bash
MODEL_PATH="Qwen2.5-VL-7B-Instruct"
MODEL_PATH="/mnt/bn/datasave-lf3-forsave/data-save/models/Qwen2.5-VL-7B-Instruct"
# MODEL_PATH="/mnt/bn/datasave-lf3-forsave/data-save/models/Qwen2.5-VL-0.5B-Instruct"
DATASET="CMIVQA"
# /mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/charades
TRAIN_DATA="/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/codes/train/train.json"
EVAL_DATA="/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/codes/test/test.json"
VIDEO_FOLDER="/mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/videos"
MAX_PIX=3584
MIN_PIX=16
NUM_WORKERS=16
NUM_WORKERS=1
OUTPUT_DIR=/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/CMIVQA_preprocessed_data_maxpix_3584_clipvideos


cd /mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero
python3 preprocess_dataset.py \
  --model_name $MODEL_PATH \
  --dataset $DATASET \
  --train_data_path $TRAIN_DATA \
  --eval_data_path $EVAL_DATA \
  --video_folder $VIDEO_FOLDER \
  --max_pix_size $MAX_PIX \
  --min_pix_size $MIN_PIX \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR
