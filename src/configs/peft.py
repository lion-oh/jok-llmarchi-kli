from dataclasses import dataclass, field
from typing import List

@dataclass
class lora_config:
    r: int=4
    lora_alpha: int=8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias="none"
    task_type: str="CAUSAL_LM"
    lora_dropout: float=0.05