from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class quantization_config:
    load_in_4bit: bool=True
    bnb_4bit_quant_type: str='nf4' # 4-bit NormalFloat Quantization
    bnb_4bit_compute_dtype: torch.dtype=field(default_factory=lambda: torch.float32)