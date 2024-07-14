from dataclasses import dataclass, field
from typing import Optional
import logging
import math
import sys
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraConfig
)

'''
Ref.

DPO
https://huggingface.co/docs/trl/main/en/dpo_trainer

MoRA
https://github.com/kongds/MoRA

FineTuning
https://wandb.ai/byyoung3/mlnews2/reports/Fine-Tuning-Llama-3-with-LoRA-TorchTune-vs-HuggingFace--Vmlldzo3NjE3NzAz
https://colab.research.google.com/drive/1or-JzMwFolyl2KAuTaOZXGwPYx6pympX?usp=sharing


'''



# import evaluate
from datasets import load_dataset
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback # peft의 경우 Trainer로 잘 저장이 안되는 이슈가 있음. // 최신버전에서는 없어졋난봄.
from tqdm import tqdm

logger = logging.getLogger(__name__)


__LOSS_FUNCTIONS__ =  ['SFT', 'DPO'] # DPO는 안되겟다.. dataset을 손봐야함.
__PROMPT_TEMPLATE__ = ['sj_1', 'hr_1', 'jh_1']
SEED = 42


import argparse

def define_argparser(
        result_save_name='../resource/result/result_jh_v1.json', 
        model_save_root_path = '../resource/model_history/',
        peft_type='MoRA',):

    __peft_type__ = ['MoRA', 'LoRA'] # LoRA++, 

    if not peft_type in __peft_type__:
        raise Exception(f"peft_type should be one of {__peft_type__}")

    p = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

    # model arguments
    p.add_argument('--model_name_or_path', default='upstage/SOLAR-10.7B-Instruct-v1.0')    # 모델 선호하는 것 있음 바꾸면되는거구               

    # model training arguments
    p.add_argument('--batch_size', default=8) # 8
    p.add_argument('--warmup_steps', default=20)
    p.add_argument("--learning_rate", default=2e-5)

    # p.add_argument('--micro_batch_size', default=16)
    p.add_argument('--num_epoch', default=10)
    p.add_argument('--logging_steps', default=16)

    # loss function
    p.add_argument("--loss_funtion", default='SFT')

    # path
    p.add_argument("--train_data_path", default='./dataset/일상대화요약_train.json')
    p.add_argument("--valid_data_path", default='./dataset/일상대화요약_test.json')
    p.add_argument("--submit_data_path", default='./dataset/일상대화요약_dev.json')
    p.add_argument("--result_submit_path", default=result_save_name)
    p.add_argument("--model_save_path", default=model_save_root_path)


    # quantization arguments   -> bits
    # p.add_argument('--load_in_bit', default='')
    p.add_argument("--load_dtype", default='bfloat16')
    '''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    '''

    # peft arguments
    p.add_argument('--peft_method', default=peft_type)
    p.add_argument('--lora_rank', default=8)
    p.add_argument("--lora_alpha", default=16)
    p.add_argument('--task_type', default='CAUSAL_LM')
    p.add_argument("--lora_dropout", default=0.05)

    
    if peft_type == 'MoRA':
        p.add_argument('--mora_type', default=6)
        p.add_argument("--lora_target_modules", default='q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj')
    
    elif peft_type == 'LoRA':
        pass
    
    else:
        raise Exception("check spelling")

 
    '''
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    '''

    config = p.parse_args()

    return config




def load_model_tokenizer(model_name_or_path, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device=device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token


if __name__ == "__main__":
    config = define_argparser()

