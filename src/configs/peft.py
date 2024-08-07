from dataclasses import dataclass, field
from typing import List

# @dataclass
# class lora_config:
#     r: int=16
#     lora_alpha: int=32
#     target_modules: List[str] = field(default_factory=lambda: ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'])
#     bias="none"
#     task_type: str="CAUSAL_LM"
#     lora_dropout: float=0.05

@dataclass
class lora_config:
    use_mora: bool = True
    mora_type: int=6
    r: int=16
    lora_alpha: int=32
    target_modules: List[str] = field(default_factory=lambda: ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'])
    bias="none"
    task_type: str="CAUSAL_LM"
    lora_dropout: float=0.05
