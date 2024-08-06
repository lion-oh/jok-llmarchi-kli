import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

from configs import train_config as TRAIN_CONFIG
from configs.datasets import base_dataset as DATASET_CONFIG
from data.dataloader import CustomDataset, DataCollatorForSupervisedDataset
from utils.config_utils import (
    update_config,
    generate_peft_config
)



def main(**kwargs):
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=train_config.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_id if train_config.tokenizer is None else train_config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = CustomDataset(DATASET_CONFIG.train_split, tokenizer)
    valid_dataset = CustomDataset(DATASET_CONFIG.valid_split, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=train_config.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.lr,
        weight_decay=0.1,
        num_train_epochs=train_config.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=train_config.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,  # CUDA 환경에서만 가능.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,  # 이슈
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
        peft_config=peft_config
    )

    trainer.train()

if __name__ == "__main__":
    # params = {
    #     "model_id": "microsoft/Phi-3-mini-4k-instruct",
    #     "target_modules": "all-linear"
    # }
    main()