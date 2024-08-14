from dataclasses import dataclass


@dataclass
class train_config:
    model_id: str="MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer: str=None
    save_dir: str="../baseline/resource/results/"
    cache_dir: str="./models/"
    batch_size: int=2
    gradient_accumulation_steps: int=2
    lr: float=2e-5
    epoch: int=5
    peft_method: str="lora"
    use_peft: bool=True
    warmup_steps: int=20


@dataclass
class train_config_jh:
    model_id: str="MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer: str=None
    save_dir: str="./finetuned_model/"
    cache_dir: str="./models/"
    tensorboard_log_path: str='./training_log/'
    batch_size: int=1
    gradient_accumulation_steps: int=16
    warmup_steps: int=20
    lr: float=2e-5
    epoch: int=10
    use_peft: bool=True
    peft_method: str="lora"
    use_quantization: bool=True
