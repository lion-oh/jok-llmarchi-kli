from warnings import warn

from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config_utils import update_config, generate_quantization_config
from src.configs.quantization import quantization_config as QUANTIZATION_CONFIG

# Function to load the main model for text generation
def load_model(model_name, quantization, cache_dir):
    if type(quantization) == type(True):
        warn("Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.", FutureWarning)
        quantization = "4bit"

    bnb_config = None
    if quantization:
        quant_config = QUANTIZATION_CONFIG()
        bnb_config = generate_quantization_config(quant_config)

    kwargs = {}
    if bnb_config:
        kwargs.update(bnb_config)

    kwargs["device_map"]="auto"
    kwargs["cache_dir"]=cache_dir
    print("model_name", model_name)
    print("kwargs", kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=False,
        **kwargs
    )
    return model

def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model