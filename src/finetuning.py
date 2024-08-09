import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_callback import TrainerCallback # peft의 경우 Trainer로 잘 저장이 안되는 이슈가 있음. // 최신버전에서는 없어졋난봄.
from transformers.integrations import TensorBoardCallback
import tensorboard

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# from configs import train_config as TRAIN_CONFIG
from configs import train_config_jh as TRAIN_CONFIG
from configs.datasets import base_dataset as DATASET_CONFIG
from configs.quantization import quantization_config as QUANTIZATION_CONFIG

from data.dataloader import (
    CustomDataset, 
    DataCollatorForSupervisedDataset
)

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_quantization_config,
)

from utils.general_utils import (
    make_training_log
)

from grokfast_pytorch.grokfast import GrokFastAdamW

def main(**kwargs):
    train_config = TRAIN_CONFIG()
    quantization_config = QUANTIZATION_CONFIG()
    update_config(train_config, **kwargs)

    quantization_config_dict = generate_quantization_config(train_config, quantization_config)

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_id,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=train_config.cache_dir,
        **quantization_config_dict,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_id if train_config.tokenizer is None else train_config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 실험별 로그 저장 경로 생성
    ROOT, training_details = make_training_log(train_config, peft_config, quantization_config)
    training_log = ROOT + training_details
    writer = SummaryWriter(log_dir=training_log)
    tensorboard_callback = TensorBoardCallback(writer)

    # 개인 환경에 맞게 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG.train_split = os.path.join(current_dir, DATASET_CONFIG.train_split)
    DATASET_CONFIG.valid_split = os.path.join(current_dir, DATASET_CONFIG.valid_split)

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
        output_dir= os.path.join(train_config.save_dir, training_details),
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
        max_seq_length=4096,  # 이슈
        packing=True,
        report_to=["tensorboard"],
        seed=42,
    )

    ## optimizer
    if train_config.use_grokfast:
        optimizer_args = {"optimizers" : (GrokFastAdamW(model.parameters()), None)}
    else:
        optimizer_args = {}

    # LOAD TRAINER
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        peft_config=peft_config,
        callbacks=[tensorboard_callback],
        **optimizer_args
    )

    trainer.train()

if __name__ == "__main__":
    # params = {
    #     "model_id": "microsoft/Phi-3-mini-4k-instruct",
    #     "target_modules": "all-linear"
    # }

    main()