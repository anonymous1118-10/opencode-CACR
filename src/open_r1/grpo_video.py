# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video as Qwen2VLGRPOTrainer


from src.open_r1.trainer import Qwen2VLGRPOVLLMTrainer_Video as Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from tqdm import tqdm
import torch
import json
import random
from reward_fun import iou_timestamp_reward_fordebert, parse_timestamp_output_debert
from deepspeed.runtime.zero.config import ZeroStageEnum
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'  # Reverts to weights_only=False
os.environ['TORCH_ALLOW_UNSAFE_GLOBALS'] = '1'  # 禁用全局变量安全检查
# Solution 1: Recommended safe approach
from deepspeed.runtime.fp16.loss_scaler import LossScaler

# Add all necessary DeepSpeed classes to safe globals
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        ZeroStageEnum,
        LossScaler,
        # Add any other classes that appear in future errors here
    ])
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
        default="",
        metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    )




def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    pattern = re.compile(r'<think>.*?</think>.*?<answer>.*?</answer>', re.DOTALL)
    # pattern = re.compile(r'<思考>.*?</思考>\s*<答案>.*?</答案>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print('matches:', matches)
    for content,data in zip(completions,matches):
        if data==None:
            print('content not match format,details:')
            print(content)
            print("---------------------------------------")
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         rank = torch.distributed.get_rank()
    # else:
    #     rank = 0  # 如果不是分布式训练，假设是主进程

    # if rank == 0:  # 只有主进程执行以下代码
    #     import pdb
    #     pdb.set_trace()  # 设置断点
    rewards=[1.0 if match else 0.0 for match in matches]
    print("format reward :",rewards)
    return [1.0 if match else 0.0 for match in matches]

# iou_timestamp_reward 
reward_funcs_registry = {
    "iou": iou_timestamp_reward_fordebert, # Modified registry to use iou_timestamp_reward_fordebert
    "format": format_reward,
}

QUESTION_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

def load_json_dataset(train_data_path, eval_data_path, video_folder, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r') as f:
            data = json.load(f)
        examples = []
        for video_id, video_data in tqdm(data.items()):
            for sentence_id, (timestamps, sentence) in enumerate(zip(video_data['timestamps'], video_data['sentences'])):
                sentence = sentence.strip().lower()
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
                    logger.warning(f"Video file not found for ID: {video_id}")
                    continue
                
                example = {
                    "problem": sentence,
                    "solution": (timestamps[0], timestamps[1]),
                    "video_path": video_path,
                    "durations": video_data['duration'],
                    "preprocessed_path": "" # Initialize preprocessed_path as None
                }
                # if  video_data['duration']>300:
                #     continue
                if preprocessed_data_path != "": # If preprocessed data path is provided, construct the path
                    example["preprocessed_path"] = os.path.join(preprocessed_data_path, split_name, f"{video_id}_{sentence_id}")
                examples.append(example)
        random.seed(42)
        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        logger.info(f"Loaded {len(examples)} examples for {split_name} split")#1573
        
        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx): # Define getitem within the scope where dataset is available
            example = dataset[idx]

            # return example
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            # print(data_to_return)
            # print("preprocessed_path:", example["preprocessed_path"])
            if example["preprocessed_path"] != "": # Check if preprocessed path exists
                try:
                    # data_to_return["image_inputs"] = [torch.load(os.path.join(example["preprocessed_path"][0], "image_inputs.pt"))]
                    data_to_return["video_inputs"] = [torch.load(os.path.join(example["preprocessed_path"][0], "video_inputs.pt"))]
                    # data_to_return["video_inputs"] = [torch.load(os.path.join(example["preprocessed_path"][0], "extract_frames_with_subtitles.pt"))]
                    # extract_frames_with_subtitles.pt
                    with open(os.path.join(example["preprocessed_path"][0], "video_kwargs.json"), 'r') as f:
                        data_to_return["video_kwargs"] = [json.load(f)]
                    data_to_return["use_preprocessed"] = [True] # Flag to indicate preprocessed data is used
                    print(f"success: loading preprocessed data from {example['preprocessed_path'][0]}, back to video_path.")
                except Exception as e:
                    print(f"Warning: Error loading preprocessed data from {example['preprocessed_path'][0]}, falling back to video_path. Error: {e}")
                    pdb.set_trace()
                    data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
            else:
                data_to_return["use_preprocessed"] = [False] #  No preprocessed data to use or path invalid

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    
    # 加载整个数据集字典
    dataset = load_from_disk(script_args.dataset_name)
    
    def read_data(dataset):
                # 查看第一个训练样本
        sample = dataset['train'][0]  # 如果是字典结构；如果是列表则直接 dataset[0]

        # 打印样本所有键和对应数据类型/形状
        print("=== Sample keys and data types ===")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: tensor of shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, (list, np.ndarray)):
                print(f"{key}: {type(value)} of length/shape {len(value) if isinstance(value, list) else value.shape}")
            else:
                print(f"{key}: {type(value)}:{value}")

        # 打印视频数据的具体信息（假设键为'pixel_values_videos'）
        if 'pixel_values_videos' in sample:
            video_data = sample['pixel_values_videos']
            print("\n=== Video tensor details ===")
            print(f"Shape: {video_data.shape}")
            print(f"Min value: {video_data.min().item()}")
            print(f"Max value: {video_data.max().item()}")
            print(f"Mean value: {video_data.float().mean().item()}")

        # 打印其他关键信息（如文本/标签）
        if 'text' in sample:
            print(f"\nText: {sample['text']}")
        if 'label' in sample:
            print(f"Label: {sample['label']}")
    read_data(dataset)
    # Format into conversation
 
    try:
        trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
        training_args.resume_from_checkpoint = model_args.model_name_or_path
        # trainer_cls=v1_Qwen2VLGRPOTrainer 
        print("using: ", trainer_cls)#using:  <class 'src.open_r1.trainer.grpo_trainer_video.Qwen2VLGRPOTrainer_Video'>
        
        logger.info("Initializing training process")
        logger.info(f"Training arguments: {training_args}")
        # pdb.set_trace()
        # Initialize the GRPO trainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )

        # 在trainer初始化后
        logger.info(f"Using trainer class: {trainer_cls.__name__}")
        logger.info("Trainer initialized successfully")
        logger.info(f"Initialized trainer with model: {model_args.model_name_or_path}")
        
        logger.info("Starting model training...")
        print("training_args.resume_from_checkpoint:",training_args.resume_from_checkpoint)
        # Train and push the model to the Hub
        # 替换 trainer.train()
        # 有checkpoint 从checkpoint 继续训练
        # train_result = trainer.train(resume_from_checkpoint=True)
        trainer.train()
        logger.info("Training completed successfully")
        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
        logger.info("Training completed successfully")
        logger.info(f"Model saved to {training_args.output_dir}")
    except Exception as e:
        logger.critical("Critical error occurred during training", exc_info=True)
        raise
import logging
from logging import getLogger
import os

# 修改日志配置部分
log_dir = os.path.dirname(os.getenv("LOG_PATH", "./default.log"))
os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_PATH", "./training.log")),  # 使用环境变量
        logging.StreamHandler()
    ]
)
logger = getLogger(__name__)
if __name__ == "__main__":
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     rank = torch.distributed.get_rank()
    # else:
    #     rank = 0  # 如果不是分布式训练，假设是主进程

    # if rank == 0:  # 只有主进程执行以下代码
    # wandb.init(project="grpo-video-hl", name="timezero", notes="this is a test",mode="dryrun")  # 仅保存数据到本地，不启动服务)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # if rank == 0:  # 只有主进程执行以下代码
    # wandb.config.update(training_args) 
    main(script_args, training_args, model_args)
