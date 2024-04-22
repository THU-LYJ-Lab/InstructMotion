import torch
from peft import PeftModel, PeftModelForSeq2SeqLM, get_peft_model, LoraConfig, TaskType, load_peft_weights, set_peft_model_state_dict

def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model

def load_pretrained_test(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    for k, v in list(state_dict.items()):
        if "vae" not in k and "lm.language_model" not in k:
            del state_dict[k]
            state_dict["lm.language_model." + k] = v
    model.load_state_dict(state_dict, strict=False)
    return model

def load_pretrained_test_peft(cfg, model, logger=None, phase="train", r=8, lora_alpha=32, lora_dropout=0.1):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # for k, v in list(state_dict.items()):
    #     if "default.default" in k:
    #         del state_dict[k]
    #         state_dict[k.replace("default.default", "default")] = v
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=True, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model.lm.language_model = get_peft_model(model.lm.language_model, peft_config)
    missing, unexpected = model.lm.language_model.load_state_dict(state_dict, strict=True)
    return model

def load_pretrained_vae(cfg, model, logger=None):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu")['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    if hasattr(model, 'vae'):
        model.vae.load_state_dict(vae_dict, strict=True)
    else:
        model.motion_vae.load_state_dict(vae_dict, strict=True)
    
    return model
