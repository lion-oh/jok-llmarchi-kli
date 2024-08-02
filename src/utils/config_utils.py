from dataclasses import asdict

from peft import LoraConfig

from ..configs import datasets, lora_config, train_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")

def generate_peft_config(train_config, kwargs):
    configs = [lora_config]
    peft_configs = [LoraConfig]
    print(lora_config.__name__.rstrip("_config"))
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config
