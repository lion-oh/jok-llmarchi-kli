from dataclasses import dataclass, field
from functools import partial
from typing import Optional
import logging
import json
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

from trl import SFTTrainer, SFTConfig

# data
from preprocess import (
    CustomDataset,
    torchdataset_generator,
    DataCollatorForSupervisedDataset,
    collate_fn,
)

from datasets import Dataset, Features, Value

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
import argparse

logger = logging.getLogger(__name__)


__LOSS_FUNCTIONS__ =  ['SFT', 'DPO'] # DPO는 안되겟다.. dataset을 손봐야함.
__PROMPT_TEMPLATE__ = ['sj_1', 'hr_1', 'jh_1']
SEED = 42






def define_argparser(
        result_save_name='./result/inference/result_jh_v1.json', 
        model_save_root_path = './result/model_history/',
        peft_type='MoRA',):

    __peft_type__ = ['MoRA', 'LoRA'] # LoRA++, 

    if not peft_type in __peft_type__:
        raise Exception(f"peft_type should be one of {__peft_type__}")

    p = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

    # model arguments
    p.add_argument('--model_name_or_path', default='upstage/SOLAR-10.7B-Instruct-v1.0')    # 모델 선호하는 것 있음 바꾸면되는거구               

    # model training arguments
    p.add_argument('--batch_size', default=8) # 8
    # p.add_argument("--gradient_accumulation_steps", type=int, default=64,                                                               help="gradient accumulation steps")
    p.add_argument('--warmup_steps', default=10)
    p.add_argument("--learning_rate", default=2e-5)
    p.add_argument("--gradient_accumulateion_steps", default=1)

    # p.add_argument('--micro_batch_size', default=16)
    p.add_argument('--num_epoch', default=10)
    p.add_argument('--logging_steps', default=16)


    # loss function
    p.add_argument("--loss_funtion", default='SFT')

    # path                                       
    p.add_argument("--train_data_path", default='/data1/kaggle/Korean_DCS_2024/data/일상대화요약_train.json')
    p.add_argument("--valid_data_path", default='/data1/kaggle/Korean_DCS_2024/data/일상대화요약_test.json')
    p.add_argument("--submit_data_path", default='/data1/kaggle/Korean_DCS_2024/data/일상대화요약_dev.json')
    p.add_argument("--result_submit_path", default=result_save_name)
    p.add_argument("--model_save_path", default=model_save_root_path)


    # quantization arguments   -> bits
    # p.add_argument('--load_in_bit', default='')
    p.add_argument("--model_dtype", default='bfloat16')
    p.add_argument("--device", default='cuda')
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
    p.add_argument("--inference_mode", default=False)

    
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


    p.add_argument("--save_dir", default='./result/model_history/')




    config = p.parse_args()

    return config




def load_model_tokenizer(model_name_or_path, dtype, config, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        **kwargs
    )#.to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_float_type(config_dtype):
    if config_dtype not in ['bfloat16']:
        raise Exception("check your float type")
    
    if config_dtype == 'bfloat16':
        return torch.bfloat16


def load_peft_config(config):
    if config.peft_method == 'MoRA':
        loraconfig = LoraConfig(
            use_mora=True,
            mora_type=config.mora_type,
            r = config.lora_rank,
            # MoRA does not use lora_alpha
            # lora_alpha=lora_alpha,
            target_modules=(config.lora_target_modules).split(","),
            lora_dropout=config.lora_dropout,
            task_type='CASUAL_LM',
            inference_mode=config.inference_mode,
        )
        return loraconfig
    elif config.peft_method == 'LoRA':
        pass

def load_data(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data

if __name__ == "__main__":
    # Tensorboard 코드 작성 필요
    ## -> Loss function, metric 추적 관찰 필요
    config = define_argparser()

    # Load Model
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=config.model_name_or_path,
        dtype=load_float_type(config.model_dtype),
        config = config
    )

    # Model Arguments
    # SFT Arguments
    # Training Arguments
    

    # PeftLoRA Argument
    ## load lora config
    loraconfig = load_peft_config(config)

    # wrapping with model
    model = get_peft_model(model, loraconfig)
    if config.peft_method == "MoRA":
        model = model.merge_and_unload() 

    # dataset
    train_data = load_data(config.train_data_path)
    valid_data = load_data(config.valid_data_path)

    train_dataset = CustomDataset(
        train_data,
        tokenizer,
        config)
    valid_dataset = CustomDataset(
        valid_data,
        tokenizer,
        config
    )

    ## dpo를 사용할 거면 코드를 좀 다르게 짜야함.
    features = Features({
        "input_ids": Value("string"),
        "labels": Value("string")
    })


    train_dataset = Dataset.from_generator(
        torchdataset_generator, 
        gen_kwargs={'torch_dataset': train_dataset},
        features=features)

    valid_dataset = Dataset.from_generator(
        torchdataset_generator, 
        gen_kwargs={'torch_dataset': valid_dataset},
        features=features)
    
    # dataCollate을 사용하거나, collate fn 을 사용하거나,,,
    # batch shuffle옵션은 어디있지?
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=config.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        # gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.1,
        num_train_epochs=config.num_epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        packing=True,
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
        # data_collator=data_collator,
        # eval_strategy="epoch",
        args=training_args,
    )

    trainer.train()


    # save model
    ## trainer안에 epoch마다 save model 하는 부분이 있는지 찾아보기


    # check metrics
    ## trainer안에  ROUGUE, BLUERT? 등 메트릭 넣을 수 있는지 확인하기


    
    '''
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    '''