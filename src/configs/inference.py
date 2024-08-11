from dataclasses import dataclass, field


@dataclass
class infer_config:
    model_id: str = field(
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B", metadata={"help": "finetuning model id"}
    )
    tokenizer: str=None
    output_dir: str="../baseline/resource/outputs/"
    cache_dir: str="./models/"
    peft_id: str=None
    quantization: str=None
    max_new_tokens: int=1024
    seed: int=42
    do_sample: bool=False

