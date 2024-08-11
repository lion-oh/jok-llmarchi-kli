import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import fire

import json
import tqdm

import torch
from accelerate.utils import is_xpu_available
from transformers import AutoTokenizer

from configs.inference import infer_config as INFER_CONFIG
from configs.datasets import base_dataset as DATASET_CONFIG
from data.dataloader import CustomDataset
from utils.model_utils import load_model, load_peft_model
from utils.config_utils import update_config




def main(**kwargs):
    infer_config = INFER_CONFIG()
    update_config(infer_config, **kwargs)

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(infer_config.seed)
    else:
        torch.cuda.manual_seed(infer_config.seed)
    torch.manual_seed(infer_config.seed)

    model = load_model(infer_config.model_id, infer_config.quantization, infer_config.cache_dir, **kwargs)
    if infer_config.peft_id:
        model = load_peft_model(model, infer_config.peft_id)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(infer_config.model_id if infer_config.tokenizer is None else infer_config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CustomDataset(DATASET_CONFIG.test_split, tokenizer)
    with open(DATASET_CONFIG.test_split, "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to("cuda").unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

    with open(infer_config.output_dir, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    fire.Fire(main())