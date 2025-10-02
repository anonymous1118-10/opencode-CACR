#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
# 路径配置
BASE_DIR="/mnt/bn/hl-multidata-save/data-save/code/TimeZero"
cd $BASE_DIR

# 模型路径数组
MODEL_BASES=(
    # "/mnt/bn/hl-multidata-save/data-save/models/Qwen2.5-VL-7B-Instruct"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-1600"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-1800"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-2000"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-1000"
)



# 数据集路径
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/caption-pre/VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated'
VIDEO_FOLDER='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/VPTSL/VehicleVQA/car/videos'

# 计算总样本数
total_samples=$(python -c "from datasets import load_from_disk; ds = load_from_disk('$DATASET_PATH')['test']; print(len(ds))")
echo "Total evaluation samples: $total_samples"

# 任务块大小
TASK_SIZE=100

# 遍历每个模型进行评测
for MODEL_BASE in "${MODEL_BASES[@]}"; do
    CHECKPOINT_DIR="${BASE_DIR}/ceval-result/modeforall/$(basename $(dirname "${MODEL_BASE%/}"))/$(basename "${MODEL_BASE%/}_7B-Instruct_VehicleVQA_better")"
    mkdir -p "$CHECKPOINT_DIR"
    
    # 创建任务队列
    TASK_QUEUE="$CHECKPOINT_DIR/task_queue.txt"
    seq 0 $TASK_SIZE $total_samples > "$TASK_QUEUE"
    echo "$total_samples" >> "$TASK_QUEUE"  # 添加结束标记
    
    # 创建锁文件
    LOCK_FILE="$CHECKPOINT_DIR/task_queue.lock"
    touch "$LOCK_FILE"
    
    # 创建日志目录
    LOG_DIR="${CHECKPOINT_DIR}/logs"
    mkdir -p "$LOG_DIR"
    chmod 777 "$LOG_DIR"
    MASTER_LOG="${LOG_DIR}/job_master.log"
    touch "$MASTER_LOG"
    chmod 666 "$MASTER_LOG"

    # 获取GPU列表
    gpu_list="0 1 2 3 4 5 6 7"
    num_gpus=$(echo $gpu_list | wc -w)
    
    # 工作函数：处理任务
    process_tasks() {
        local gpu_id=$1
        local log_file="${LOG_DIR}/gpu${gpu_id}.log"
        
        while true; do
            # 获取下一个任务块
            (
                flock -x 200
                start_index=$(head -n 1 "$TASK_QUEUE")
                if [ -z "$start_index" ]; then
                    echo "GPU $gpu_id: No more tasks" >> "$MASTER_LOG"
                    exit 0
                fi
                
                # 计算结束索引
                end_index=$((start_index + TASK_SIZE))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi
                
                # 更新任务队列
                tail -n +2 "$TASK_QUEUE" > "${TASK_QUEUE}.tmp"
                mv "${TASK_QUEUE}.tmp" "$TASK_QUEUE"
            ) 200>"$LOCK_FILE"
            
            # 检查是否所有任务已完成
            if [ $start_index -ge $total_samples ]; then
                echo "GPU $gpu_id: All tasks completed" >> "$MASTER_LOG"
                break
            fi
            
            # 执行评估任务
            echo "GPU $gpu_id processing: $start_index-$end_index" >> "$MASTER_LOG"
            {
                echo "===== TASK START [$(date +%Y-%m-%d_%H:%M:%S)] ====="
                torchrun --nproc_per_node=1 \
                         --master_port=$((11113 + RANDOM % 10000)) \
                         eval_ondebert-all.py \
                            --dataset_path "$DATASET_PATH" \
                            --model_base "$MODEL_BASE" \
                            --dataset CMIVQA  \
                            --checkpoint_dir "$CHECKPOINT_DIR" \
                            --batch_size 2 \
                            --device "cuda:${gpu_id}" \
                            --start_index $start_index \
                            --end_index $end_index \
                            --video_dir "$VIDEO_FOLDER"
                echo "===== TASK END [$(date +%Y-%m-%d_%H:%M:%S)] ====="
            } >> "$log_file" 2>&1
        done
    }
    
    # 启动所有GPU工作进程
    for gpu_id in $gpu_list; do
        process_tasks $gpu_id &
    done
    
    # 等待所有后台任务完成
    wait
    echo "Evaluation for model $MODEL_BASE completed."
done