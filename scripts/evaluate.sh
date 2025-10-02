
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_IB_DISABLE=1  # 禁用 InfiniBand，使用 TCP/IP
export NCCL_SOCKET_IFNAME=eth0  # 使用 eth0 作为通信接口
export NCCL_P2P_DISABLE=1  # 禁用 P2P 通信
export NCCL_DEBUG=INFO  # 启用 NCCL 调试模式，获取详细日志
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
# 其他环境变量
cd /mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero
# MODEL_BASE=mllm/Qwen2.5-VL-7B-Instruct
MODEL_BASE=./output/cmivqa_outputs_video_trainerori/checkpoint-600
MODEL_BASE=./output/cmivqa_outputs_video_trainerv1_20250511_fullframes/checkpoint-200
MODEL_BASE=./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_v1/checkpoint-300
# 提前创建目录
CHECKPOINT_DIR="ceval-result/$(basename $(dirname "${MODEL_BASE%/}"))/$(basename "${MODEL_BASE%/}")"
# mkdir -p "$CHECKPOINT_DIR"evaluate_for_ngpu_v0.py
# 运行命令（注意 --checkpoint_dir 只出现一次）941
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12333 evaluate_for_ngpu_v1.py \
--model_base $MODEL_BASE \
--dataset CMIVQA \
--checkpoint_dir "$CHECKPOINT_DIR" \
--batch_size 2 \
--device cuda:0 \
--start_index 0 \
--end_index 600

# /mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/output/cmivqa_outputs_video_Qwen2.5-VL-7B-Instruct
# python
# torchrun --nproc_per_node="1" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12301" \
#     evaluate_gpu.py \
#      --model_base $MODEL_BASE \
#      --dataset charades \
#      --checkpoint_dir ckpt_charades
# echo “will save in ceval-result/$(basename $(dirname "${MODEL_BASE%/}"))/$(basename "${MODEL_BASE%/}")”

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12939 evaluate.py --model_base $MODEL_BASE --dataset CMIVQA --checkpoint_dir --checkpoint_dir "ceval-result/$(basename $(dirname "${MODEL_BASE%/}"))/$(basename "${MODEL_BASE%/}")"
# ceval-result/orickpt_CMIVQA-600-debug