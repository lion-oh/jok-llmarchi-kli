
import logging
import json
import math
import sys
import os
from tqdm import tqdm
import argparse
import pandas as pd

from preprocess import (
    CustomDataset,
    torchdataset_generator,
    DataCollatorForSupervisedDataset,
    # collate_fn,
    Collate,
    lambda_unsqeeze,
    PreprocessData,
    preprocess_to_tokenize
)

import torch
import torch.nn as nn

from datasets import Dataset, Features, Value, Sequence
# from dataclasses import dataclass, field

from peft import (
    # get_peft_config,
    get_peft_model,
    # get_peft_model_state_dict,
    # set_peft_model_state_dict,
    TaskType,
    # PeftType,
    # PrefixTuningConfig,
    # PromptEncoderConfig,
    LoraConfig
)

from trl import SFTTrainer, SFTConfig

import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    # AutoConfig, 
    AutoModelForCausalLM,
    # HfArgumentParser, 
    # Trainer, 
    # TrainingArguments, 
    # DataCollatorForLanguageModeling
)

from prompt import jh_prompt_template

# from functools import partial
# from typing import Optional


from transformers.trainer_callback import TrainerCallback # peft의 경우 Trainer로 잘 저장이 안되는 이슈가 있음. // 최신버전에서는 없어졋난봄.


logger = logging.getLogger(__name__)


__LOSS_FUNCTIONS__ =  ['SFT', 'DPO'] # DPO는 안되겟다.. dataset을 손봐야함.
__PEFT_TYPE__ = ['MoRA', 'LoRA']
# __PROMPT_TEMPLATE__ = ['sj_1', 'hr_1', 'jh_1']
SEED = 42



def define_argparser(
        result_save_name='./result/inference/result_jh_v1.json', 
        model_save_root_path = './result/model_history/',
        peft_type='MoRA',):

    if not peft_type in __PEFT_TYPE__:
        raise Exception(f"peft_type should be one of {__PEFT_TYPE__}")

    p = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

    # model arguments
    p.add_argument('--model_name_or_path', default='beomi/Llama-3-Open-Ko-8B')    # 모델 선호하는 것 있음 바꾸면되는거구        // upstage/SOLAR-10.7B-Instruct-v1.0        
    
    # BitsAndBytesConfig으로 바꿔야함.
    p.add_argument("--model_bitsize", default='4bit')
    p.add_argument("--max_length", default=4096)

    # model training arguments
    p.add_argument('--batch_size', default=1) # 8
    p.add_argument("--gradient_accumulation_steps", type=int, default=16,  help="gradient accumulation steps")
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



def load_model_tokenizer(model_name_or_path, config, **kwargs):
    dic = {}
    if config.model_bitsize == '8bit':
        dic['load_in_8bit'] = True
    elif config.model_bitsize == '4bit':
        dic['load_in_4bit'] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # torch_dtype=dtype,
        device_map='auto',
        **dic
    )#.to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

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
            task_type=TaskType.CAUSAL_LM,
            inference_mode=config.inference_mode,
        )
    elif config.peft_method == 'LoRA':
        ############################################################ config 추가 ###############################
        loraconfig = LoraConfig(
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            inference_mode=config.inference_mode,
        )

    return loraconfig

def wrapping_model_with_peft(model, loraconfig, precise_layernorm=False, precise_final_layer=True):

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if precise_layernorm:
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    if precise_final_layer:
        model.lm_head = CastOutputToFloat(model.lm_head)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model = get_peft_model(model, loraconfig)

    print("-"*100 + 'trainable_paramameters' + '-'*100)
    print_trainable_parameters(model)
    print("-"*300)
    return model


if __name__ == '__main__':
    import pprint
    
    config = define_argparser(peft_type='LoRA')
    print("-"*100 + 'config' + '-'*100)
    pprint.pprint(config)
    print("-"*300)


    model, tokenizer = load_model_tokenizer(
        model_name_or_path = config.model_name_or_path,
        config = config
    )

    loraconfig = load_peft_config(config)

    model = wrapping_model_with_peft(
        model = model,
        loraconfig = loraconfig)
    

    # data process
    train = PreprocessData(config.train_data_path)
    valid = PreprocessData(config.valid_data_path)

    train_df = train.make_columns()
    valid_df = valid.make_columns()


    train_dataset = preprocess_to_tokenize(
        train_df = train_df,
        tokenizer= tokenizer,
        object_prompt=jh_prompt_template.OBJECT_PROMPT_1,
        system_prompt=jh_prompt_template.SYSTEM_PROMPT_1,
        config=config
    )

    valid_dataset = preprocess_to_tokenize(
        train_df = valid_df,
        tokenizer= tokenizer,
        object_prompt=jh_prompt_template.OBJECT_PROMPT_1,
        system_prompt=jh_prompt_template.SYSTEM_PROMPT_1,
        config=config
    )


    training_args = SFTConfig(
        output_dir=config.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        # bf16=True,
        fp16=True,
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
        # data_collator=partial(collate_fn, tokenizer=tokenizer),
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # eval_strategy="epoch",
        args=training_args,
        peft_config=loraconfig,
    )

    trainer.train()