# coding: utf-8
import os
import json
import torch
import argparse
from datasets import Dataset, DatasetDict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import decord
from moviepy.editor import VideoFileClip
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video dataset for Qwen-VL model")
    parser.add_argument("--model_name", type=str, default="/mnt/bn/datasave-lf3-forsave/data-save/models/Qwen2.5-VL-7B-Instruct",
                        help="Path to the pretrained model")
    parser.add_argument("--dataset", type=str, default="CMIVQA",
                        help="Dataset name to be preprocessed")
    parser.add_argument("--train_data_path", type=str, default="/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/codes/train/train.json",
                        help="Path to the training data JSON file")
    parser.add_argument("--eval_data_path", type=str, default="/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/codes/test/test.json",
                        help="Path to the evaluation data JSON file")
    parser.add_argument("--video_folder", type=str, default="/mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/videos",
                        help="Path to the folder containing video files")
    parser.add_argument("--output_dir", type=str, default='/mnt/bn/datasave-lf3-forsave/data-save/code/TimeZero/data_prepare/CMIVQA_preprocessed_data_maxpix_3584_clipvideos',
                        help="Output directory path. If None, it will be created based on dataset and max_pix values")
    parser.add_argument("--max_pix_size", type=int, default=3584,
                        help="Maximum pixel size")
    parser.add_argument("--min_pix_size", type=int, default=16,
                        help="Minimum pixel size")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes for multiprocessing")
    
    return parser.parse_args()

def preprocess_single_video(task_args): # Accept task arguments as a single tuple/list
    video_path, processor, max_pixels, min_pixels, example_output_dir, sentence, solution, duration = task_args # Unpack task args
    try:
        # image_inputs, video_inputs, video_kwargs, fps_inputs = preprocess_video_inner(
        #     video_path, processor, max_pixels, min_pixels
        # )

        image_inputs, video_inputs, video_kwargs, fps_inputs =preprocess_video_inner_spans(video_path,solution, duration,processor, max_pixels, min_pixels)

        os.makedirs(example_output_dir, exist_ok=True)

        # torch.save(image_inputs, os.path.join(example_output_dir, "image_inputs.pt"))
        torch.save(video_inputs, os.path.join(example_output_dir, "video_inputs.pt"))
        with open(os.path.join(example_output_dir, "video_kwargs.json"), 'w') as f:
            json.dump(video_kwargs, f)

        return {
            "problem": sentence,
            "solution": solution,
            "preprocessed_path": example_output_dir,
            "duration": duration,
            "status": "success"
        }
    except Exception as e:
        print(f"Warning: Preprocessing failed for video {video_path}, skipping. Error: {e}")
        return {
            "video_path": video_path,
            "status": "failed",
            "error": str(e)
        }
def extract_st_ed_subtitles(subtitle_path,id,starttime,end_time):
    with open(subtitle_path, 'r') as file:
        subtitles = json.load(file)
        # subtitles=sub_file
    extract_datas=[]
    data=subtitles[id] if id in subtitles.keys() else None
    if data is None:
        return None
    start_list=[]
    end_list=[]
    for subtitle in data:
        start = subtitle.get('start', 0)
        end = subtitle.get('end', float('inf'))
        if (starttime>=start and starttime<=end) or (end_time>=start and end_time<=end) or  (start>=starttime and end<=end_time):
            extract_datas.append(subtitle)
            start_list.append(start)
            end_list.append(end)
    return extract_datas,start_list,end_list

def preprocess_video_inner_spans(video_path,solution, duration,processor,max_pixels,min_pixels):
    start_time=solution[0]*duration
    end_time=solution[1]*duration
    extract_datas,start_list,end_list=extract_st_ed_subtitles('/mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/subtitle.json',name_id,start_time,end_time)
    clip_start,clip_end=start_list[0],end_list[-1]
    print("clip_start",clip_start)
    print("clip_end",clip_end)
    pdb.set_trace()
    
    # 生成新视频的保存路径
    video_filename = os.path.basename(video_path)
    video_name, ext = os.path.splitext(video_filename)
    clip_dir = "/mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/clip_videos"
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, f"{video_name}_clip_{start_time:.2f}_{end_time:.2f}{ext}")
    # 剪辑视频并保存
    try:
        with VideoFileClip(video_path) as video:
            if end_time > video.duration:
                end_time = video.duration  # 确保不超过原视频长度
            if start_time >= end_time:
                raise ValueError(f"Start time ({start_time}) must be less than end time ({end_time})")
                
            clip = video.subclip(start_time, end_time)
            clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
            clip.close()
        print("start_time:",start_time,"end_time:",end_time)
        print(f"视频剪辑成功，保存至: {clip_path}")
    except Exception as e:
        print(f"视频剪辑失败: {e}")
        # 出错时使用原始视频
        clip_path = video_path
    # import pdb; pdb.set_trace()
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": clip_path, 
                "total_pixels": max_pixels, 
                "min_pixels": min_pixels,
                },
            ]
        },
    ]
    


    # import pdb;pdb.set_trace()
    # return None,None,None,None,
    
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    return image_inputs, video_inputs, video_kwargs, fps_inputs
def preprocess_video_inner(video_path, processor, max_pixels, min_pixels):
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": video_path, 
                "total_pixels": max_pixels, 
                "min_pixels": min_pixels,
                },
            ]
        },
    ]
    import pdb; pdb.set_trace()
    
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    return image_inputs, video_inputs, video_kwargs, fps_inputs
def preprocess_video_inner_subtitle(video_path, subtitle_entries, max_pixels, min_pixels):
    name_id=video_path.split('/')[-1].split('.')[0]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    if not os.access(video_path, os.R_OK):
        
        raise PermissionError(f"无读取权限: {video_path}")
        
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    try:
        vr = decord.VideoReader(video_path)
        print(f"VideoReader initialized successfully: {vr}")
    except Exception as e:
        print(f"Error initializing VideoReader: {e}")
        return None
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    total_duration = total_frames / fps  # 视频总时长（秒）

    # 处理截取时间范围
    start_frame = 0
    end_frame = total_frames - 1
    if video_start is not None:
        start_frame = min(int(video_start * fps), total_frames - 1)
    else:
        start_frame=0
        video_start=0
    if video_end is not None:
        end_frame = min(int(video_end * fps), total_frames - 1)
    else:
        end_frame=total_frames - 1
        video_end=end_frame/fps
    # 提取有效时间点并转换为帧号
    frame_indices = []
    valid_entries = []
    for entry in subtitle_entries:
        t = entry["start"]
        # 时间点需在有效范围内
        if (video_start is not None and t < video_start) or (video_end is not None and t > video_end):
            continue
        # 计算相对帧号
        relative_t = t - (video_start if video_start else 0)
        frame_num = int(relative_t * fps) + start_frame
        # 确保帧号在合法区间
        frame_num = max(start_frame, min(frame_num, end_frame))
        frame_indices.append(frame_num)
        valid_entries.append(entry)
    
    # 批量读取帧
    if not frame_indices:
        return []
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # 转换为PIL图像并与字幕配对
    frame_subtitle_pairs = []
    for idx, frame in enumerate(frames):
        # pdb.set_trace()
        frame = np.transpose(frame, (2, 0, 1))
        img=torch.from_numpy(frame)
        entry = valid_entries[idx]
        frame_subtitle_pairs.append( (img, entry["text"], entry["start"]) )
    
        return frame_subtitle_pairs
def process_split(file_path, split_name, video_folder, output_dir, max_pixels, min_pixels, processor, num_workers=8):
    output_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(output_split_dir, exist_ok=True)

    with open(file_path, 'r') as f:
        data = json.load(f)  

    examples = []
    tasks = []

    for video_id, video_data in data.items():
        for sentence_id, (timestamps, sentence) in enumerate(zip(video_data['timestamps'], video_data['sentences'])):
            try:
                sentence = sentence.strip().lower()
            except:
                import pdb; pdb.set_trace()
            if sentence.endswith("."):
                sentence = sentence[:-1]

            video_filename_base = video_id
            video_path = None
            for ext in ['mp4', 'mkv', 'webm']:
                candidate_path = os.path.join(video_folder, f"{video_filename_base}.{ext}")
                if os.path.isfile(candidate_path):
                    video_path = candidate_path
                    break
            if video_path is None:
                print(f"Warning: Video file not found for ID: {video_id}")
                continue

            example_output_dir = os.path.join(output_split_dir, f"{video_id}_{sentence_id}")
            solution = (timestamps[0] / video_data['duration'], timestamps[1] / video_data['duration'])
            duration = video_data['duration']

            tasks.append((video_path, processor, max_pixels, min_pixels, example_output_dir, sentence, solution, duration)) # Prepare task arguments as tuples

    pbar = tqdm(total=len(tasks), desc=f"Preprocessing {split_name} split") # Initialize progress bar in main process

    with mp.Pool(processes=num_workers) as pool:

        results = pool.imap_unordered(preprocess_single_video, tasks) # Use imap_unordered for unordered results, potentially faster

        successful_examples = []
        failed_count = 0
        for result in results: # Iterate through results to update progress bar
            pbar.update(1)
            if result['status'] == 'success':
                successful_examples.append(result)
            else:
                failed_count += 1
                # Optionally log failed videos and errors

    pbar.close() # Close progress bar after processing

    print(f"Preprocessing for split '{split_name}' finished. Failed videos: {failed_count}, Successful videos: {len(successful_examples)}")

    return Dataset.from_list(successful_examples)


def preprocess_dataset_and_save(train_data_path, eval_data_path, video_folder, output_dir, max_pixels, min_pixels, num_workers=8):

    # processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print('MODEL_NAME', MODEL_NAME)
    print('num_workers',num_workers)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=os.path.dirname(MODEL_NAME))
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = process_split(train_data_path, "train", video_folder, output_dir, max_pixels, min_pixels, processor, num_workers)
    eval_dataset = process_split(eval_data_path, "eval", video_folder, output_dir, max_pixels, min_pixels, processor, num_workers)
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    
    # Calculate pixel values 3584*28*28=2809856
    max_pixels = args.max_pix_size * 28 * 28
    min_pixels = args.min_pix_size * 28 * 28
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = f"./{args.dataset}_preprocessed_data_maxpix_{args.max_pix_size}"
    else:
        output_dir = args.output_dir
        
    print('output_dir', output_dir)

    dataset_dict = preprocess_dataset_and_save(
        args.train_data_path, args.eval_data_path, args.video_folder, 
        output_dir, max_pixels, min_pixels, num_workers=args.num_workers
    )
    
    print("Preprocessing complete. Datasets saved to:", output_dir)
    print(dataset_dict)