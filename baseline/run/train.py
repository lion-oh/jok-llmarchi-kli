import argparse
import os
import yaml

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

from baseline.src.dataloader import CustomDataset, DataCollatorForSupervisedDataset


def load_config(config_path):
    # 절대 경로 얻기
    abs_config_path = os.path.abspath(config_path)

    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"Configuration file not found: {abs_config_path}")

    with open(abs_config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config):
    # LOAD MODEL
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_id'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir='../models'
    )

    if config['model']['tokenizer'] is None:
        config['model']['tokenizer'] = config['model']['model_id']

    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])
    tokenizer.pad_token = tokenizer.eos_token

    # LOAD DATASET
    train_dataset = CustomDataset(config['data']['train_file'], tokenizer)
    valid_dataset = CustomDataset(config['data']['valid_file'], tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # SET CONFIGS
    if config.get('peft'):
        peft_configs = config.get('peft')
        if peft_configs['name'] == 'Lora':
            lora_config = LoraConfig(
                r=peft_configs['r'],
                lora_alpha=peft_configs['lora_alpha'],
                target_modules=peft_configs['target_modules'],
                lora_dropout=peft_configs['lora_dropout'],
                bias=peft_configs['bias'],
                task_type=peft_configs['task_type']
            )

            model=get_peft_model(model, lora_config)
            print(model.print_trainable_parameters())
        else:
            print(NotImplemented)
            lora_config=None

    training_args = SFTConfig(
        output_dir=config['model']['save_dir'],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['lr'],
        weight_decay=0.1,
        num_train_epochs=config['training']['epoch'],
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=config['training']['warmup_steps'],
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024, # 이슈
        packing=True,
        seed=42,
    )

    # LOAD TRAINER
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        peft_config=lora_config
    )

    trainer.train()


if __name__ == "__main__":
    '''cli
    CUDA_VISIBLE_DEVICES=1,3 python -m train --config baseline/run/configs/base.yaml
    '''
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)