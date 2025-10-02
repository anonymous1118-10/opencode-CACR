import numpy as np
import os
import re
import pdb
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import wandb
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration
# AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)
from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video as Qwen2VLGRPOTrainer

from src.open_r1.trainer import Qwen2VLGRPOVLLMTrainer_Video as Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from tqdm import tqdm
import torch
import json
import random
import logging
from logging import getLogger
import os

EPSILON = 1e-6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_PATH", "./training.log")),  # 使用环境变量
        logging.StreamHandler()
    ]
)
logger = getLogger(__name__)
def parse_timestamp_output_debert(output_string):
    """解析输出字符串中的时间戳，返回(start_time, end_time)元组"""
    # 查找所有<answer>标签内的内容
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    
    if not answer_matches:
        logger.debug(f"未找到<answer>标签: {output_string}")
        return None  # 没有找到<answer>标签
    
    # 使用最后一个<answer>标签内的内容
    last_answer_content = answer_matches[-1].strip()
    logger.debug(f"解析答案内容: {last_answer_content}")
    
    # 检查是否为"Out of video"（忽略大小写和空格）
    if re.search(r'^\s*out\s*of\s*video\s*$', last_answer_content, re.IGNORECASE):
        logger.debug("检测到'Out of video'，返回(0, 0)")
        return 0, 0
    
    # 定义时间戳模式：支持中英文"to"，以及秒的单位(s/秒)
    time_pattern = r"(\d+\.?\d*)\s*(s|秒)?\s*(to|至|and)\s*(\d+\.?\d*)\s*(s|秒)?"
    matches = re.search(time_pattern, last_answer_content, re.IGNORECASE)
    
    if not matches:
        logger.debug(f"未找到有效的时间戳格式: {last_answer_content}")
        return None
    
    # 提取开始和结束时间
    start_time = float(matches.group(1))
    end_time = float(matches.group(4))
    
    logger.debug(f"解析结果: start_time={start_time}, end_time={end_time}")
    return start_time, end_time


def iou_timestamp_reward_fordebert(completions, solution, durations, clip_solution, **kwargs):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    alpha=0.5
    alpha=1
    # 这里duration =clip_solution[1]-clip_solution[0]
    # pdb.set_trace()
    for content, sol, duration, clip_sol in zip(completions, solution, durations, clip_solution):
        try:
            # 
            reward = 0.0
            # pred 必须在clip duration 内；或者为0 才
            parsed_times = parse_timestamp_output_debert(content)
            if parsed_times is None:
                rewards.append(reward)
                continue
            
            clip_start,clip_end=clip_sol[0],clip_sol[1]
            if clip_start>clip_end:
                clip_start,clip_end=clip_end,clip_start
            # gt
            gt_start, gt_end = sol[0], sol[1]
            if gt_start > gt_end:
                gt_start, gt_end = gt_end, gt_start
            
            s, e = gt_start, gt_end
            has_overlap = max(0, min(clip_end, e) - max(clip_start, s)) > 0
            # 1.clip_solution solution没有交集 parsed_times=0,0 ->1;else 0
            if has_overlap<=0:
                if abs(parsed_times[0]) < EPSILON and abs(parsed_times[1]) < EPSILON:
                    reward = 1.0*alpha
                else:
                    reward = 0.0
            else:
                # 2.clip_solution solution有交集 parsed_times 在clip 范围 ->1;else 0
                if (0 <= parsed_times[0] <= clip_start and
                    0 <= parsed_times[1] <= clip_end):
                    start_time, end_time = parsed_times
                    from_number = start_time+clip_start
                    to_number = end_time+clip_start

                    intersection = max(0, min(to_number, e) - max(from_number, s))
                    union = max(to_number, e) - min(from_number, s)
                    if union > 0:
                        reward = intersection / union
                else:
                    reward = 0.0
            

            
            print('gt second:', gt_start, gt_end)
            print('clip second:', clip_start, clip_end)
            print('pred second:', parsed_times if parsed_times else "not found")
            print(f"------------- {current_time} IoU reward: {reward} -------------\n")
            # pdb.set_trace()
            rewards.append(reward)
            logger.debug(f"IoU reward calculated: {reward} | GT: {gt_start}-{gt_end} | Clip: {clip_start}-{clip_end} | Pred: {parsed_times if parsed_times else (0, 0)}")
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"Content: {content}\n")
                    f.write(f"pred second: {str(parsed_times if parsed_times else (0, 0))}\n")
                    f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                    f.write(f"clip second: {str(clip_start)}, {str(clip_end)}\n")
                    f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
        
        except Exception as e:
            logger.error(f"Error calculating IoU reward: {str(e)}", exc_info=True)
            rewards.append(0.0)
            continue
    # pdb.set_trace()
    return rewards

def parse_timestamp_output_debert(output_string):
    """解析输出字符串中的时间戳，返回(start_time, end_time)元组"""
    # 查找所有<answer>标签内的内容
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    
    if not answer_matches:
        logger.debug(f"未找到<answer>标签: {output_string}")
        return None  # 没有找到<answer>标签
    
    # 使用最后一个<answer>标签内的内容
    last_answer_content = answer_matches[-1].strip()
    logger.debug(f"解析答案内容: {last_answer_content}")
    
    # 检查是否为"Out of video"（忽略大小写和空格）
    if re.search(r'^\s*out\s*of\s*video\s*$', last_answer_content, re.IGNORECASE):
        logger.debug("检测到'Out of video'，返回(0, 0)")
        return 0, 0
    
    # 定义时间戳模式：支持中英文"to"，以及秒的单位(s/秒)
    time_pattern = r"(\d+\.?\d*)\s*(s|秒)?\s*(to|至|and)\s*(\d+\.?\d*)\s*(s|秒)?"
    matches = re.search(time_pattern, last_answer_content, re.IGNORECASE)
    
    if not matches:
        logger.debug(f"未找到有效的时间戳格式: {last_answer_content}")
        return None
    
    # 提取开始和结束时间（修正索引）
    start_time = float(matches.group(1))
    end_time = float(matches.group(4))  # 修正：使用第4个捕获组获取结束时间数值
    
    logger.debug(f"解析结果: start_time={start_time}, end_time={end_time}")
    return start_time, end_time
