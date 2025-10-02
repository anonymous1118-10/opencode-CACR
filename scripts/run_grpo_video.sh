
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 添加内存碎片整理配置
export PYTHONPATH=".:$PYTHONPATH"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand，使用 TCP/IP
export NCCL_SOCKET_IFNAME=eth0  # 使用 eth0 作为通信接口
export NCCL_P2P_DISABLE=1  # 禁用 P2P 通信
export NCCL_DEBUG=INFO  # 启用 NCCL 调试模式，获取详细日志
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
dataset_path='TutorialVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_43.45_0719_full_dataset_textclipcaptions_updated'

# | 0/1550 [00:00<?, ?it/s]====compute_loss=====5/12390 
DATASET_NAME=$(basename "$dataset_path")
CURRENT_DATE=$(date +%Y%m%d)
# run_grpo_video_cmivqa_clipvideo_v1
OUTDIR=output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_${TIMESTAMP}
OUTDIR=output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_${DATASET_NAME}_topkclips
OUTDIR=output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_${DATASET_NAME}_onlycaption
OUTDIR=output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_${DATASET_NAME}_onlypresult
mkdir -p "$OUTDIR"
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_IFNAME=eth0  # 替换为实际网卡名 12333
export DEBUG_MODE="true"
mkdir -p $OUTDIR/log
sudo chmod 777 $OUTDIR/log
export LOG_PATH="$OUTDIR/log/run_grpo_video_cmivqa_clipvideo_v1_${TIMESTAMP}.txt"  # 添加时间戳
PRINT_DIR="$OUTDIR/print"
mkdir -p "$PRINT_DIR"



cd /mnt/bn/hl-multidata-save/data-save/code/CACR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12378" \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path  /mnt/bn/hl-multidata-save/data-save/models/Qwen2.5-VL-7B-Instruct \
    --preprocessed_data_path data_prepare/CMIVQA_preprocessed_data_maxpix_3584_clipvideos \
    --train_data_path data_prepare/codes/train/train.json \
    --eval_data_path data_prepare/codes/test/test.json \
    --video_folder /mnt/bn/hl-multidata-save/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/videos \
    --dataset_name $dataset_path \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $WANDB_NAME \
    --report_to none \
    --save_steps 50 \
    --save_only_model False \
    --resume_from_checkpoint True \
    2>&1 | tee -a "$PRINT_DIR/training_log_${TIMESTAMP}.txt"
   