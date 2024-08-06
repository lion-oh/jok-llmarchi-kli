

def make_training_log(config, peft_config=None, quantization_config=None):
    from datetime import datetime
    today_date = datetime.today()
    date_str = today_date.strftime("%Y-%m-%d")
    
    root = config.tensorboard_log_path
    batch_size = int(config.batch_size)*int(config.gradient_accumulation_steps)
    epoch = int(config.epoch)
    peft_method = str(config.peft_method)

    main_parameters = f'batch_size({batch_size})' + '_' +\
        f'epoch({epoch})' + '_' + \
        f'peft_method({peft_method})'
    
    if config.use_peft:
        r = peft_config.r
        alpha = peft_config.lora_alpha
        target_modules = str(peft_config.target_modules)

        peft_parameters = f'r({r})' + '_' + \
            f'alpha({alpha})' + '_' + \
            f'target_modules({target_modules})'
    else:
        peft_parameters = 'NoPeft'

    if config.use_quantization:
        if getattr(quantization_config, 'load_in_4bit'):
            load_bit = '4bit'
            comput_type = str(quantization_config.bnb_4bit_compute_dtype)
        elif getattr(quantization_config, 'load_in_8bit'):
            load_bit = '8bit'
            comput_type = str(quantization_config.bnb_8bit_compute_dtype)
        else:
            raise Exception("Check your quantization type")
        quant_parameters = f'load_bit({load_bit})' + '_' +\
            f'compute_type({comput_type})'
    else:
        quant_parameters = 'NoQuantization'

    return root, main_parameters + '_' + peft_parameters + '_' + quant_parameters