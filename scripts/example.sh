
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 添加内存碎片整理配置
export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_IFNAME=eth0  # 替换为实际网卡名
export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_tg_video.txt"
cd /mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path /mnt/bn/datasave-lf3-forsave/data-save/models/Qwen2.5-VL-7B-Instruct \
    --preprocessed_data_path /mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/Charades_preprocessed_data_maxpix_3584 \
    --train_data_path ./Charades/charades_annotation/train.json \
    --eval_data_path ./Charades/charades_annotation/val.json \
    --video_folder /mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/Charades/Charades_v1 \
    --dataset_name xxx \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to none \
    --save_steps 50 \
    --save_only_model true  
    # > timezero_output_exp.log 2>&1
    # > timezero_output.log 2>&1
# sdpa flash_attention_2