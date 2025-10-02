#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# 路径配置
BASE_DIR="/mnt/bn/hl-multidata-save/data-save/code/TimeZero"
cd /mnt/bn/hl-multidata-save/data-save/code/TimeZero
# /cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-150
# 定义多个模型路径
# total step 690 traindata 2213 test data 122 med
MODEL_BASES=(
"./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-400"
"./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-800"
"./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-1050"
"./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-1000"
"./output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset_timezero/checkpoint-600"

)
MODEL_BASES=(
    # "/mnt/bn/hl-multidata-save/data-save/models/Qwen2.5-VL-7B-Instruct"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-1600"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-1800"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-2100"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-800"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-2700"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-700"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_topkclips/checkpoint-2900"
)

MODEL_BASES=(
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlycaption/checkpoint-3800"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlycaption/checkpoint-3200"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlycaption/checkpoint-3600"
)
MODEL_BASES=(
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlypresult/checkpoint-800"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlypresult/checkpoint-1600"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlypresult/checkpoint-2000"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlypresult/checkpoint-3000"
    "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlypresult/checkpoint-2500"
)
# "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlycaption/checkpoint-1700"
    # "/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated_onlycaption/checkpoint-1000"
   
# DATASET_PATH='/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/CMIVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert/datadict/dataset_dict_clipcaption_pre_result'
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/subtitles/MedVidQA_preprocessed_data_maxpix_3584_subtitle_0721_11_full_dataset_clean'
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/subtitles/TutorialVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset'
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/subtitles/VehicleVQA_preprocessed_data_maxpix_3584_subtitle_full_dataset' # | 47/1860 372/epoch
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/caption-pre/VehicleVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_83.39_0716_fulldata_textclipcaption_textclipcaptions_updated'
VIDEO_FOLDER='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/VPTSL/TutorialVQA/tutorial/videos'
VIDEO_FOLDER='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/VPTSL/VehicleVQA/car/videos'
total_samples=$(python -c "from datasets import load_from_disk; ds = load_from_disk('$DATASET_PATH')['test']; print(len(ds))")
# total_samples=16
gpu_list="0 1 2 3 4 5 6 7"  # 使用空格分隔的字符串代替数组
# gpu_list="0 1 2 3"  # 使用空格分隔的字符串代替数组
# gpu_list="0 1"  # 使用空格分隔的字符串代替数组
# gpu_list="0 5 6 7"  # 使 用空格分隔的字符串代替数组
num_gpus=$(echo $gpu_list | wc -w)
per_gpu_samples=$(( ($total_samples + $num_gpus - 1) / $num_gpus ))

# 遍历每个模型进行评测
for MODEL_BASE in "${MODEL_BASES[@]}"; do
    CHECKPOINT_DIR="${BASE_DIR}/ceval-result/modeforall/$(basename $(dirname "${MODEL_BASE%/}"))/$(basename "${MODEL_BASE%/}_add00")"
    mkdir -p "$CHECKPOINT_DIR"
    # 创建日志目录并设置权限
    LOG_DIR="${CHECKPOINT_DIR}/logs"
    mkdir -p "$LOG_DIR"
    chmod 777 "$LOG_DIR"  # 确保有写入权限
    # 主日志文件路径
    MASTER_LOG="${LOG_DIR}/job_master.log"
    touch "$MASTER_LOG"
    chmod 666 "$MASTER_LOG"

    echo "Total evaluation samples: $total_samples"    
    echo "Using $num_gpus GPUs, $per_gpu_samples samples per GPU"
    echo "Evaluating model: $MODEL_BASE"

    #  启动并行任务
    i=0
    for gpu_id in $gpu_list; do
        (
            start=$((i * per_gpu_samples))
            end=$(( (i + 1) * per_gpu_samples ))
            [ $end -gt $total_samples ] && end=$total_samples
            port=$((11113 + i))
            
            # 生成日志路径
            timestamp=$(date +%Y%m%d_%H%M%S)
            log_file="${LOG_DIR}/gpu${gpu_id}_${start}-${end}_${timestamp}.log"
            
            echo "[MASTER] Starting GPU ${gpu_id}: samples ${start}-${end}" | tee -a "$MASTER_LOG"
            
            # 执行命令并双写日志
            {
                echo "===== TASK START [$(date +%Y-%m-%d_%H:%M:%S)] ====="
                torchrun --nproc_per_node=1 \
                         --nnodes=1 \
                         --node_rank=0 \
                         --master_addr="localhost" \
                         --master_port=$port \
                    evel_ondebert-all.py \
                        --dataset_path "$DATASET_PATH" \
                        --model_base "$MODEL_BASE" \
                        --dataset CMIVQA  \
                        --checkpoint_dir "$CHECKPOINT_DIR" \
                        --batch_size 2 \
                        --device "cuda:${gpu_id}" \
                        --start_index $start \
                        --end_index $end \
                        --video_dir "$VIDEO_FOLDER"
                        # --pred_mode 1 \
                        

                echo "===== TASK END [$(date +%Y-%m-%d_%H:%M:%S)] ====="
            } 2>&1 | tee -a "$log_file" "$MASTER_LOG"
            
            echo "[MASTER] GPU ${gpu_id} completed" | tee -a "$MASTER_LOG"
        ) &
        i=$((i + 1))
    done

    wait
    echo "Evaluation for model $MODEL_BASE completed."
done