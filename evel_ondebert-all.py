from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from moviepy import VideoFileClip
import os
import re
import pickle
import torch
from datasets import load_from_disk
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import pdb

VIDEO_INFO_CACHE = {}
DATASET_PATH='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/datadict/caption-pre/TutorialVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_43.45_0719_full_dataset_textclipcaptions_updated'
VIDEO_FOLDER='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/VPTSL/TutorialVQA/tutorial/videos'

# clip_video_dir = '/mnt/bn/datasave-lf3-forsave/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/clip_videos_ondebert'
clip_video_dir='/mnt/bn/hl-multidata-save/data-save/train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/clip_videos_ondebert'
def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    parser.add_argument('--dataset', default='CMIVQA', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/mnt/bn/qmg-datasave-hl/workspace/VPTSL/eval-models/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_TutorialVQA_preprocessed_data_maxpix_3584_clipvideos_ondebert_miou_43.45_0719_full_dataset_textclipcaptions_topkclips/checkpoint-1600")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="/mnt/bn/hl-multidata-save/data-save/code/TimeZero/output/cmivqa_outputs_video_trainer_cmivqa_clipvideo_ondebertv0_alpha0-5_clipcaption_cmedvidqa-dataset_dict_clipcaption_textclipcaptions_updated_onlypresult/checkpoint-450")
    parser.add_argument("--resume", default='False', help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to use")
    parser.add_argument("--start_index", type=int, default=0, help="start_index")
    parser.add_argument("--end_index", type=int, default=1000, help="--end_index")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="--dataset_path")
    parser.add_argument("--video_dir", type=str, default=VIDEO_FOLDER, help="--video_dir")
    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def cached_process_vision_info(messages, return_video_kwargs=False):
    global VIDEO_INFO_CACHE

    video_path = None
    for msg in messages:
        for content in msg.get('content', []):
            if isinstance(content, dict) and 'video' in content:
                video_path = content['video']
                break

    cache_key = f"{video_path}_{return_video_kwargs}"
    if cache_key in VIDEO_INFO_CACHE:
        return VIDEO_INFO_CACHE[cache_key]

    result = process_vision_info(messages, return_video_kwargs=return_video_kwargs)
    VIDEO_INFO_CACHE[cache_key] = result

    return result

def make_prompt_text(x):
    prompt_text = """<|im_start|>system
    You are a video analysis assistant.<|im_end|>
    <|im_start|>user
    To accurately pinpoint the event "{problem}" in the video, determine the precise time period of the event.
    Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
    Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".
    If the video does not contain the answer to the question, please output "0 to 0".
    This video introduces """.format(problem=x["problem"])
    
    vision_part = [
        x['clip_caption'],
        "<|vision_start|>",
        "<|video_pad|>",
        "<|vision_end|>"
    ]
    
    full_prompt = prompt_text + "\n".join(vision_part) + "<|im_end|>\n<|im_start|>assistant\n"
    return full_prompt.replace('\n',' ')
def make_prompt_text_caption(x):
    prompt_text = """<|im_start|>system
    You are a video analysis assistant.<|im_end|>
    <|im_start|>user
    To accurately pinpoint the event "{problem}" in the video, determine the precise time period of the event.
    Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
    Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".
    If the video does not contain the answer to the question, please output "0 to 0".
    This video introduces """.format(problem=x["problem"])
    
    vision_part = [
        x['clip_caption'],
        "<|vision_start|>",
        "<|video_pad|>",
        "<|vision_end|>"
    ]
    
    full_prompt = prompt_text + "\n".join(vision_part) + "<|im_end|>\n<|im_start|>assistant\n"
    return full_prompt.replace('\n',' ')


def load_preprocessed_features(data):
    """加载预处理特征，如果可用"""
    path = data.get("preprocessed_path", "")
    if not path:
        return None, None, False
    
    # 检查视频特征文件
    video_inputs_pt_path = os.path.join(path, "video_inputs.pt")
    if not os.path.exists(video_inputs_pt_path):
        return None, None, False
    
    # 检查FPS配置文件
    fps_json_path = os.path.join(path, "video_kwargs.json")
    if not os.path.exists(fps_json_path):
        return None, None, False
    
    try:
        # 加载视频特征
        video_inputs = torch.load(video_inputs_pt_path)
        
        # 加载FPS配置
        with open(fps_json_path, 'r') as f:
            fps_dict = json.load(f)
            fps_inputs = fps_dict.get("fps", [2.0])  # 默认为2.0
        
        # 确保特征可用
        if video_inputs is not None and fps_inputs is not None:
            return video_inputs, fps_inputs, True
    
    except Exception as e:
        print(f"Error loading preprocessed features: {e}")
    
    return None, None, False

def inference(video_path, data, model, processor, max_new_tokens=2048, device="cuda:0",presult=False):
    # 首先尝试加载预处理特征
    video_inputs, fps_inputs, features_loaded = load_preprocessed_features(data)
    features_loaded=False
    image_inputs = None
    print(data['id'])
    print('video_path',video_path)
    # pdb.set_trace()
    if video_path is None or not os.path.exists(video_path):
        return None, False
    # pdb.set_trace()
    if features_loaded:
        # 使用预处理特征
        text = make_prompt_text(data)
        print("使用预处理特征")
    else:
        # 使用原始视频
        sentence = data['problem']
        prompt = GROUND_TEMPLATE.replace('[EVENT]', sentence)
        prompt+="This video introduces "+data['clip_caption']
        if presult:
            print("use problem_pre_result")
            sentence = data['problem_pre_result']
            prompt = GROUND_TEMPLATE.replace('[EVENT]', sentence)
        # pdb.set_trace()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path,
                "total_pixels": 3584 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
                },
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)
        fps_inputs = video_kwargs['fps'] if 'fps' in video_kwargs else [2.0]
        # pdb.set_trace()
    # 准备模型输入
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True,
                       return_tensors="pt")
    inputs = inputs.to(device)

    # 执行推理
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 解码输出
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0], features_loaded

def parse_timestamp_output(output_string):
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content)
            if answer_matches:
                last_match = answer_matches[-1]
                return float(last_match[0]), float(last_match[2])
        return None, None

    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]

    try:
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        return start_time, end_time
    except ValueError:
        return None, None
def parse_timestamp_output_debert(output_string):
    """解析输出字符串中的时间戳，返回(start_time, end_time)元组"""
    # 查找所有<answer>标签内的内容
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    
    if not answer_matches:
        print(f"未找到<answer>标签: {output_string}")
        return None, None  # 没有找到<answer>标签
    
    # 使用最后一个<answer>标签内的内容
    last_answer_content = answer_matches[-1].strip()
    print(f"解析答案内容: {last_answer_content}")
    
    # 检查是否为"Out of video"（忽略大小写和空格）
    if re.search(r'^\s*out\s*of\s*video\s*$', last_answer_content, re.IGNORECASE):
        print("检测到'Out of video'，返回(0, 0)")
        return 0, 0
    
    # 定义时间戳模式：支持中英文"to"，以及秒的单位(s/秒)
    time_pattern = r"(\d+\.?\d*)\s*(s|秒)?\s*(to|至|and)\s*(\d+\.?\d*)\s*(s|秒)?"
    matches = re.search(time_pattern, last_answer_content, re.IGNORECASE)
    
    if not matches:
        print(f"未找到有效的时间戳格式: {last_answer_content}")
        return None,None
    
    # 提取开始和结束时间
    start_time = float(matches.group(1))
    end_time = float(matches.group(4))
    
    print(f"解析结果: start_time={start_time}, end_time={end_time}")
    return start_time, end_time
GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".If the video does not contain the answer to the question, please output "0 to 0"."""

def create_work_items(data):
    work_items = list(data)
    # random.shuffle(work_items)  # Optional: shuffle items for randomness
    return work_items

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    print(f"Loading model from {model_base}")
    # model_path = f"file://{model_base}"
    model_path=model_base
    # pdb.set_trace()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

def get_checkpoint_path(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pkl")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {'processed_items': set(), 'ious': [], 'recall': np.array([0, 0, 0])}

def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)

def process_work_items(work_items, video_dir_path, model_base, device, checkpoint_dir, resume=False, args=None):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    processed_data = []
    missing_path = []
    
    # 检查是否存在已保存的完整结果
    existing_results = {}
    sorted_merged_result_path = os.path.join(checkpoint_dir, 'sorted_merged_result_sorted.json')
    sorted_merged_result_path=''
    if os.path.exists(sorted_merged_result_path):
        print(f"Loading existing results from {sorted_merged_result_path}")
        try:
            with open(sorted_merged_result_path, 'r', encoding='utf-8') as f:
                existing_results_list = json.load(f)
                # 创建ID到结果的映射
                for res in existing_results_list:
                    existing_results[res['id']] = res
            print(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # 分离需要处理的新数据项
    new_work_items = []
    for item in work_items:
        item_id = item['id']
        if item_id in existing_results:
            # 使用已有结果
            result = existing_results[item_id]
            processed_data.append(result)
            
            # 更新指标
            if 'iou' in result and isinstance(result['iou'], (int, float)):
                iou_value = result['iou']
                ious.append(iou_value)
                recall += (thresh <= iou_value)
        else:
            # 需要处理的新数据项
            new_work_items.append(item)
    
    print(f"Total items: {len(work_items)}, Existing results: {len(processed_data)}, New items: {len(new_work_items)}")
    
    if new_work_items:
        # 设置模型
        model, processor = setup_model(model_base, device)
        
        # 处理新数据项
        pbar = tqdm(new_work_items, desc="Processing new items")
        for item in pbar:
            item_id = item['id']
            # try:
            if 1:
                # 提取必要信息
                sentence = item['problem']
                
                # 检查是否有剪辑解决方案
                if 'clip_solution' not in item or len(item['clip_solution']) < 2:
                    print(f"Skipping {item_id}: Missing clip_solution")
                    continue
                    
                pred_start_second = float(item['clip_solution'][0])
                pred_end_second = float(item['clip_solution'][1])
                
                # 尝试加载预处理特征
                # _, _, features_loaded = load_preprocessed_features(item)
                features_loaded=False
                video_path = None
                clip_path = None
                skip_processing = False
                # pdb.set_trace()
                # 如果没有预处理特征，尝试查找视频文件
                if not features_loaded:
                    # 解析视频名称
                    pos = item_id.rfind('_', 0, item_id.rfind('_'))
                    video_name = item_id[0:pos] if pos != -1 else item_id
                    print("video_name",video_name)
                    # 查找视频文件
                    video_extensions = ["mp4", "mkv", "webm", "avi", "mov", "flv", "wmv", "ts"]
                    for ext in video_extensions:
                        candidate_path = os.path.join(args.video_dir, f"{video_name}.{ext}")
                        if os.path.exists(candidate_path):
                            video_path = candidate_path
                            break
                    # pdb.set_trace()
                    if not video_path:
                        print(f"Video not found for {video_name}")
                        missing_path.append(video_name)
                        skip_processing = True
                    else:
                        # 创建剪辑路径和目录
                        clip_path = os.path.join(clip_video_dir, f"{video_name}_clip_{pred_start_second:.2f}_{pred_end_second:.2f}.mp4")
                        os.makedirs(os.path.dirname(clip_path), exist_ok=True)
                        
                        # 创建剪辑视频（如果不存在）
                        if not os.path.exists(clip_path):
                            try:
                                with VideoFileClip(video_path) as video:
                                    clip_start, clip_end = pred_start_second, pred_end_second
                                    if clip_end > video.duration:
                                        clip_end = video.duration
                                    if clip_start >= clip_end:
                                        clip_start = max(0, clip_end - 1.0)  # 确保有效时长
                                    
                                    clip = video.subclipped(clip_start, clip_end)
                                    clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
                                    clip.close()
                                print(f"Created clip: {clip_path}")
                            except Exception as e:
                                print(f"Error creating clip: {e}")
                                clip_path = video_path  # 回退到原始视频
                presult=False
                # pdb.set_trace()
                if 'onlypresult' in model_base:
                    presult=True
                ans, features_used = inference(
                    clip_path if clip_path else video_path, 
                    item, 
                    model, 
                    processor, 
                    device=device,presult=presult
                )
                if ans is None:
                    ori_sp, ori_ep=0,0
                else:
                    ori_sp, ori_ep = parse_timestamp_output(ans)
                print("parse_timestamp_output:",ori_sp, ori_ep)
                try:
                    debert_sp, debert_ep=parse_timestamp_output_debert(ans)
                except:
                    debert_sp, debert_ep=0,0
                print("parse_timestamp_output_debert:",debert_sp, debert_ep)

                # 处理解析失败的情况
                if ori_sp is None or ori_ep is None:
                    ori_sp, ori_ep = 0, 0
                if debert_sp is None or debert_ep is None:
                    debert_sp, debert_ep = 0, 0
                # 转换为绝对时间
                ori_sp = max(0, ori_sp)
                ori_ep = max(0, ori_ep)
                
                sp = ori_sp + pred_start_second
                ep = ori_ep + pred_start_second
                # !!! 注意：这里的sp,ep如相同就得符合cli_solution
                if abs(sp-ep)<1e-6:
                    sp, ep=pred_start_second,pred_end_second
                debert_sp = debert_sp + pred_start_second
                debert_ep = debert_ep + pred_start_second
                # 计算指标
                s, e = item['solution'][0], item['solution'][1]
                iou_ = calc_iou(np.array([[sp, ep]]), np.array([s, e]))[0]
                iou_ = max(iou_, 0)
                
                # 计算原始IoU作为参考
                ori_iou = calc_iou(np.array([[pred_start_second, pred_end_second]]), np.array([s, e]))[0]
                ori_iou = max(ori_iou, 0)
                
                # 更新指标
                ious.append(iou_)
                recall += (thresh <= iou_)
                print(f"sp: {sp}, ep: {ep}, debert_sp: {debert_sp}, debert_ep: {debert_ep}, iou_: {iou_}, ori_iou: {ori_iou}")
                # 存储结果

                print(f'videopath {video_path} ans:{ans}')
                result = {
                    "id": item_id,
                    "problem": sentence,
                    'solution': [s, e],
                    'clip_solution': [pred_start_second, pred_end_second],
                    "clip_path": clip_path,
                    "predstart": sp,
                    "predend": ep,
                    "debert_predstart": debert_sp,
                    "debert_predend": debert_ep,
                    "ori_iou": ori_iou,
                    "iou": iou_,
                    "features_used": features_used,
                    'ans':ans,
                    "status": "success"
                }
                processed_data.append(result)
                
                # 更新进度条
                if ious:
                    miou = sum(ious) / len(ious)
                    recall_str = f"[{recall[0]/len(ious):.4f}, {recall[1]/len(ious):.4f}, {recall[2]/len(ious):.4f}]"
                    pbar.set_postfix({"mIoU": f"{miou:.4f}", 'recall': recall_str})
                if len(processed_data)%100==0:
                    start_index, end_index = args.start_index, args.end_index
                    detailed_file = f'{checkpoint_dir}/detailed_results_{start_index}_{end_index}_{len(processed_data)}.json'
                    with open(detailed_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=4)
                # pdb.set_trace()
            # except Exception as e:
            #     print(f"Error processing {item_id}: {e}")
            #     # 创建错误结果
            #     result = {
            #         "id": item_id,
            #         "problem": item.get('problem', ''),
            #         'solution': item.get('solution', [0, 0]),
            #         'clip_solution': item.get('clip_solution', [0, 0]),
            #         "clip_path": None,
            #         "predstart": 0,
            #         "predend": 0,
            #         "debert_predstart": None,
            #         "debert_predend": None,
            #         "ori_iou": 0,
            #         "iou": 0,
            #         "features_used": False,
            #         'ans':None,
            #         "status": f"error: {str(e)}"
            #     }
            #     processed_data.append(result)
    
    # 合并新结果和已有结果
    all_results = processed_data
    
    # 保存最终结果
    start_index, end_index = args.start_index, args.end_index
    result_file = os.path.join(checkpoint_dir, f"final_result_{start_index}_{end_index}.txt")
    with open(result_file, 'w') as f:
        f.write('=== FINAL RESULTS ===\n')
        f.write(f'Processed items: {len(all_results)}\n')
        
        if ious:
            mIoU = sum(ious) / len(ious)
            f.write(f'mIoU: {mIoU:.4f}\n')
            for th, r in zip(thresh, recall):
                recall_value = r / len(ious)
                f.write(f'R@{th}: {recall_value:.4f}\n')
        else:
            f.write('No valid samples processed\n')
    
    # 保存详细结果
    detailed_file = f'{checkpoint_dir}/detailed_results_{start_index}_{end_index}.json'
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    # 更新合并结果文件
    try:
        # 创建ID到结果的映射
        merged_dict = {}
        
        # 加载现有的完整结果（如果有）
        if os.path.exists(sorted_merged_result_path):
            with open(sorted_merged_result_path, 'r', encoding='utf-8') as f:
                existing_merged = json.load(f)
                for res in existing_merged:
                    merged_dict[res['id']] = res
        
        # 添加新结果（覆盖旧结果）
        for res in all_results:
            merged_dict[res['id']] = res
        
        # 转换为列表
        merged_results = list(merged_dict.values())
        
        # 按ID排序
        merged_results.sort(key=lambda x: x['id'])
        
        # 保存合并后的结果
        with open(sorted_merged_result_path, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=4)
        
        print(f"Updated merged results file with {len(merged_results)} items")
    except Exception as e:
        print(f"Error updating merged results file: {e}")
    
    # 保存缺失路径
    with open(f'{checkpoint_dir}/missing_paths_{start_index}_{end_index}.json', 'w', encoding='utf-8') as f:
        json.dump(missing_path, f, ensure_ascii=False, indent=4)
    
    print('\n=== FINAL RESULTS ===')
    if ious:
        print(f'mIoU: {sum(ious)/len(ious):.4f}')
        for th, r in zip(thresh, recall):
            print(f'R@{th}: {r/len(ious):.4f}')
    else:
        print('No valid samples processed')
    
    print(f"Results saved in: {checkpoint_dir}")
    return ious, recall

def evaluate(data, args):
    work_items = create_work_items(data)
    ious, recall = process_work_items(
        work_items,
        args.video_dir,
        args.model_base,
        args.device,
        args.checkpoint_dir,
        args.resume,
        args
    )
    return ious, recall

if __name__ == '__main__':
    args = get_args()
    
    # 加载数据集
    # if 'CMIVQA' in args.dataset:
    if 1:
        dataset = load_from_disk(args.dataset_path)
        data_split = 'test' if 'test' in dataset else 'eval'
        datasum = dataset[data_split]
        print("datasum",datasum)
        # 选择子集
        args.end_index = min(args.end_index, len(datasum))
        args.start_index = max(0, args.start_index)
        data = datasum.select(range(args.start_index, args.end_index))
        
        print(f'Evaluating {args.dataset} | Samples: {args.start_index}-{args.end_index} | Total: {len(data)}')
    # else:
    #     raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 运行评估
    evaluate(data, args)