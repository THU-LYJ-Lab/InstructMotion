import json
import os
import sys
import logging
from collections import OrderedDict
from typing import Tuple
from functools import partial 

import torch
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from datasets import Dataset
from transformers import logging as transformers_logging
from transformers import T5ForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase




"""
Contains the following utilities function for reward modeling and ppo:

    1. Create a tokenizer for motion vocab: `get_motion_tokenizer`
    2. Loading pre-trained motionGPT for sequence classification: `load_pretrained_reward`
    3. Returns a Dataset object from the jsonl containing labels: `load_dataset_from_jsonl`
    4. Callable to get the format for RewardTrainer: `preprocess_function`
    5. Sanity check for dataset: `write_dataset_ids`, `check_ids_match`


"""


logger = transformers_logging.get_logger("transformers")

def get_colorful_handler() -> logging.Handler:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        # %(green)s[%(asctime)s](%(filename)s %(lineno)d): %(white)s%(message)s
        "%(log_color)s[%(asctime)s](%(filename)s %(lineno)d): %(white)s%(message)s",
        # "%(log_color)s%(levelname)-8s%(reset)s %(green)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )
    ch.setFormatter(formatter)
    return ch

def get_motion_tokenizer(
    tokenizer_name: str,
    codebook_size: int
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=True)
    codebook_size_with_start_end_mask_tokens = codebook_size + 3
    new_vocab = [f'<motion_id_{i}>' for i in range(codebook_size_with_start_end_mask_tokens)]
    tokenizer.add_tokens(new_vocab)
    
    return tokenizer


def load_pretrained_reward_rm(model: torch.nn.Module, ckpt_path: str):
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "language_model" in k:
            if ".shared" in k:
                name = k.replace("lm.language_model.shared", "transformer.shared")
                new_state_dict[name] = v
            elif ".encoder" in k:
                name = k.replace("lm.language_model.encoder", "transformer.encoder")
                new_state_dict[name] = v
            elif ".decoder" in k:
                name = k.replace("lm.language_model.decoder", "transformer.decoder")
                new_state_dict[name] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")
    return model
    

def load_pretrained_reward_ppo(model: torch.nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "shared" in k:
            name_idx = k.find("shared")
            name = "pretrained_model." + k[name_idx:]
            new_state_dict[name] = v
        elif "encoder" in k:
            name_idx = k.find("encoder")
            name = "pretrained_model." + k[name_idx:]
            new_state_dict[name] = v
        elif "decoder" in k:
            name_idx = k.find("decoder")
            name = "pretrained_model." + k[name_idx:]
            new_state_dict[name] = v
        elif "lm_head" in k:
            name_idx = k.find("lm_head")
            name = "pretrained_model." + k[name_idx:]
            new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")
    return model

def load_pretrained_reward_dpo(model: torch.nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "shared" in k:
            name_idx = k.find("shared")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "encoder" in k:
            name_idx = k.find("encoder")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "decoder" in k:
            name_idx = k.find("decoder")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "lm_head" in k:
            name_idx = k.find("lm_head")
            name = k[name_idx:]
            new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")
    return model
    
def load_pretrained_reward_ppo_value(model: torch.nn.Module, ckpt_path: str):
    
    reward_model = T5ForSequenceClassification.from_pretrained(ckpt_path)
    
    state_dict = reward_model.state_dict()
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "transformer.shared" in k:
            name = k.replace("transformer.shared", "pretrained_model.shared")
            new_state_dict[name] = v
        elif "transformer.encoder" in k:
            name = k.replace("transformer.encoder", "pretrained_model.encoder")
            new_state_dict[name] = v
        elif "transformer.decoder" in k:
            name = k.replace("transformer.decoder", "pretrained_model.decoder")
            new_state_dict[name] = v
        elif "transformer.lm_head" in k:
            name = k.replace("transformer.lm_head", "pretrained_model.lm_head")
            new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")

    return model

def load_pretrained_reward_ppo_peft(model: torch.nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "shared" in k:
            name_idx = k.find("shared")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "encoder" in k:
            name_idx = k.find("encoder")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "decoder" in k:
            name_idx = k.find("decoder")
            name = k[name_idx:]
            new_state_dict[name] = v
        elif "lm_head" in k:
            name_idx = k.find("lm_head")
            name = k[name_idx:]
            new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")
    return model

def load_pretrained_reward_ppo_value_peft(model: torch.nn.Module, ckpt_path: str):
    
    reward_model = T5ForSequenceClassification.from_pretrained(ckpt_path)
    
    state_dict = reward_model.state_dict()
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # replace all shared with transformer.shared
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "transformer.shared" in k:
            name = k.replace("transformer.shared", "shared")
            new_state_dict[name] = v
        elif "transformer.encoder" in k:
            name = k.replace("transformer.encoder", "encoder")
            new_state_dict[name] = v
        elif "transformer.decoder" in k:
            name = k.replace("transformer.decoder", "decoder")
            new_state_dict[name] = v
        elif "transformer.lm_head" in k:
            name = k.replace("transformer.lm_head", "lm_head")
            new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"missing keys: {missing}")

    return model


def load_pretrained_vae(model: torch.nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

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
    model.load_state_dict(vae_dict, strict=True)
    
    return model

def _write_dataset_ids(dataset: Dataset, root_path: str, is_train: bool = True, use_skipped: bool = False):
    if use_skipped:
        file_name = "train_with_skipped.txt" if is_train else "eval_with_skipped.txt"
    else:
        file_name = "train.txt" if is_train else "eval.txt"
    with open(os.path.join(root_path, file_name), 'w') as f:
        for item in dataset:
            f.write("%s\n" % item["id"])

def _check_ids_match(train_dataset, root_path: str, eval_dataset, use_skipped: bool = False):
    train_file_name = "train_with_skipped.txt" if use_skipped else "train.txt"
    eval_file_name = "eval_with_skipped.txt" if use_skipped else "eval.txt"
    # Read IDs from txt files
    with open(os.path.join(root_path, train_file_name), 'r') as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(root_path, eval_file_name), 'r') as f:
        eval_ids = f.read().splitlines()

    # Extract IDs from datasets
    train_dataset_ids = [str(item['id']) for item in train_dataset]
    eval_dataset_ids = [str(item['id']) for item in eval_dataset]

    # Check if IDs match
    return train_ids == train_dataset_ids and eval_ids == eval_dataset_ids

def _preprocess_rm(
    examples, 
    tokenizer, 
    codebook_size: int, 
    use_margin: bool,
    margin_type: str,
    base_dir,
):
    """
        Examples (assumes batched):
        {
            "ID": [1, ...], 
            "prompt": ["text", ...],
            "sample_1": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 1234
                },
                ...
            ]
                
            "sample_2": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 5678
                },
                ...
            ]
            "chosen": [
                [{ 
                    "choice": "sample_i", 
                    "degree of preference": "null",
                    "user": "Matt"
                }],	
                ...
            ] 
        }
    """
    
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "label": []
    }
    if use_margin:
        new_examples["margin"] = []
        
    tokenize_with_truncation = partial(
        tokenizer, 
        padding='max_length', 
        truncation=True, # FIXME adding this result in NAN
        max_length=256, 
        add_special_tokens=True,
        return_attention_mask=True,
        # , return_tensors="pt"
    )
    def load_pairwise_string(
        prompt: str,
        sample_1: dict, 
        sample_2: dict, 
        choice: str
    ) -> Tuple[str, str]:
        def load_motion_token(sample):
            motion_token_path = sample["features"][:-4].replace('data2', 'data3') + "_token.npy"
            motion_token_path = motion_token_path.replace("features", "tokens")
            motion_token_path = os.path.join(base_dir, motion_token_path)
            return np.load(motion_token_path)[0]
        motion_token_to_string = lambda motion_token: f'<motion_id_{codebook_size}>' + ''.join([f'<motion_id_{int(i)}>' for i in motion_token.tolist()]) + f'<motion_id_{codebook_size + 1}>'
        add_prompt_string = lambda input, output: input + ' \n ' + output
        
        sample_1_token = load_motion_token(sample_1)
        sample_2_token = load_motion_token(sample_2)
        pref_motion_string = motion_token_to_string(sample_1_token if choice == "sample_1" else sample_2_token)
        dispref_motion_string = motion_token_to_string(sample_2_token if choice == "sample_1" else sample_1_token)
        
        return (
            add_prompt_string(prompt, pref_motion_string),
            add_prompt_string(prompt, dispref_motion_string)
        )
    
    for prompt, sample_1, sample_2, choice in zip(examples["prompt"], examples["sample_1"], examples["sample_2"], examples["chosen"]):
        
        chosen, rejected = load_pairwise_string(prompt, sample_1, sample_2, choice[-1]['choice'])
        tokenized_chosen = tokenize_with_truncation(chosen)
        tokenized_rejected = tokenize_with_truncation(rejected)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        # select last one because there's multiple annotations per sample, the last one is the most recent
        is_unsure_or_skipped = choice[-1]["degree of preference"] in ["Negligibly better/unsure", "skipped"]
        if is_unsure_or_skipped:
            new_examples["label"].append(torch.tensor(0.5))
        else:
            new_examples["label"].append(torch.tensor(1.0))
        if use_margin:
            if choice[-1]["degree of preference"] == "Much better":
                margin = 3. if margin_type == 'big' else 1
            elif choice[-1]["degree of preference"] == "Better":
                margin = 2. if margin_type == 'big' else 0.666
            elif choice[-1]["degree of preference"] == "Slightly better":
                margin = 1. if margin_type == 'big' else 0.333
            elif choice[-1]["degree of preference"] == "Negligibly better/unsure":
                margin = 0.
            new_examples["margin"].append(margin)
            # margin of 0 for the unsure ones
    return new_examples


def _load_dataset_from_jsonl(
    jsonl_path: str,
) -> Dataset:
    """Loads a jsonl huggingface dataset by turning it into a pandas dataframe.

    Args:
        jsonl_path (str): path to jsonl file to be converted into a dataset.

    Returns:
        Dataset: each data point is a preference.
    """
    data = list()
    with open(jsonl_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def _preprocess_ppo(examples: dict, tokenizer: PreTrainedTokenizerBase):
    """
        Examples (assumes batched):
        {
            "ID": [1, ...], 
            "prompt": ["text", ...],
            "sample_1": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 1234
                },
                ...
            ]
                
            "sample_2": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 5678
                },
                ...
            ]
            "chosen": [
                [{ 
                    "choice": "sample_i", 
                    "degree of preference": "null",
                    "user": "Matt"
                }],	
                ...
            ] 
        }
    """
    
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    
    for prompt in examples["prompt"]:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        tokenized = tokenizer.encode(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
        new_examples["query"].append(prompt)
        new_examples["input_ids"].append(tokenized)

    return new_examples

def _preprocess_dpo(
    examples, 
    tokenizer, 
    codebook_size: int, 
    use_label_smoothing,
    base_dir,
):
    """
        Examples (assumes batched):
        {
            "ID": [1, ...], 
            "prompt": ["text", ...],
            "sample_1": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 1234
                },
                ...
            ]
                
            "sample_2": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 5678
                },
                ...
            ]
            "chosen": [
                [{ 
                    "choice": "sample_i", 
                    "degree of preference": "null",
                    "user": "Matt"
                }],	
                ...
            ] 
        }
    """
    
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "label_smoothing": []
    }
        
    def load_pairwise_string(
        prompt: str,
        sample_1: dict, 
        sample_2: dict, 
        choice: str
    ) -> Tuple[str, str]:
        def load_motion_token(sample):
            motion_token_path = sample["features"][:-4].replace('data2', 'data3') + "_token.npy"
            motion_token_path = motion_token_path.replace("features", "tokens")
            motion_token_path = os.path.join(base_dir, motion_token_path)
            return np.load(motion_token_path)[0]
        motion_token_to_string = lambda motion_token: f'<motion_id_{codebook_size}>' + ''.join([f'<motion_id_{int(i)}>' for i in motion_token.tolist()]) + f'<motion_id_{codebook_size + 1}>'
        
        sample_1_token = load_motion_token(sample_1)
        sample_2_token = load_motion_token(sample_2)
        pref_motion_string = motion_token_to_string(sample_1_token if choice == "sample_1" else sample_2_token)
        dispref_motion_string = motion_token_to_string(sample_2_token if choice == "sample_1" else sample_1_token)
        
        return (
            prompt,
            pref_motion_string,
            dispref_motion_string
        )
    
    for prompt, sample_1, sample_2, choice in zip(examples["prompt"], examples["sample_1"], examples["sample_2"], examples["chosen"]):
        prompt, chosen, rejected = load_pairwise_string(prompt, sample_1, sample_2, choice[-1]['choice'])

        new_examples["prompt"].append(prompt)
        new_examples["chosen"].append(chosen)
        new_examples["rejected"].append(rejected)
        if use_label_smoothing:
            if choice[-1]["degree of preference"] == "Much better":
                label_smoothing = 0.
            elif choice[-1]["degree of preference"] == "Better":
                label_smoothing = 0.2
            elif choice[-1]["degree of preference"] == "Slightly better":
                label_smoothing = 0.3
            elif choice[-1]["degree of preference"] == "Negligibly better/unsure":
                label_smoothing = 0.5
        else:
            label_smoothing = 0.
        new_examples["label_smoothing"].append(label_smoothing)
    return new_examples


def build_rm_dataset(
    jsonl_path: str,
    root_path: str,
    tokenizer: PreTrainedTokenizerBase, 
    codebook_size: int,
    use_margin: bool,
    margin_type: str,
    sanity_check: bool = True, # checks if the dataset loaded the correct train eval samples
    use_skipped: bool = False
) -> Tuple[Dataset, Dataset]:
    dataset = _load_dataset_from_jsonl(jsonl_path)
    if not use_skipped:
        dataset = dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    if use_skipped:
        eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    
    if sanity_check:
        file_name_maybe_use_skipped = "train_with_skipped.txt" if use_skipped else "train.txt"
        if not os.path.isfile(os.path.join(root_path, file_name_maybe_use_skipped)):
            _write_dataset_ids(train_dataset, root_path, is_train=True, use_skipped=use_skipped)
            _write_dataset_ids(eval_dataset, root_path, is_train=False, use_skipped=use_skipped)
        else:
            if _check_ids_match(train_dataset, root_path, eval_dataset, use_skipped=use_skipped):
                logger.info(f"Passed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
            else:
                logger.info(f"Failed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
    
    preprocess_function_maybe_margin = partial(
        _preprocess_rm, 
        tokenizer=tokenizer, 
        codebook_size=codebook_size, 
        use_margin=use_margin, 
        margin_type=margin_type,
        base_dir=root_path
    )
    train_dataset = train_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=4,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=4,
    )
    
    return train_dataset, eval_dataset
        

def build_ppo_dataset(
    jsonl_path: str,
    seed: int,
    root_path: str,
    tokenizer: PreTrainedTokenizerBase
) -> Tuple[Dataset, Dataset]:
    dataset = _load_dataset_from_jsonl(jsonl_path)
    dataset = dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    # dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=1111)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    logger.info(_check_ids_match(train_dataset, root_path, eval_dataset))
    preprocess_ppo = partial(_preprocess_ppo, tokenizer=tokenizer)
    train_dataset = train_dataset.map(
        preprocess_ppo,
        batched=True,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_ppo,
        batched=True,
        num_proc=4
    )
    return train_dataset, eval_dataset

def build_dpo_dataset(
    use_label_smoothing: bool,
    jsonl_path: str,
    seed: int,
    root_path: str,
    tokenizer: PreTrainedTokenizerBase, 
    codebook_size: int,
    sanity_check: bool = True, # checks if the dataset loaded the correct train eval samples
    use_skipped: bool = False,
    percentage_used: float = 1.0,
    preference_type: str = "all",
    add_unsure: bool = False,
) -> Tuple[Dataset, Dataset]:
    dataset = _load_dataset_from_jsonl(jsonl_path)
    if not use_skipped:
        dataset = dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    if use_skipped:
        eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    
    if sanity_check:
        file_name_maybe_use_skipped = "train_with_skipped.txt" if use_skipped else "train.txt"
        if not os.path.isfile(os.path.join(root_path, file_name_maybe_use_skipped)):
            _write_dataset_ids(train_dataset, root_path, is_train=True, use_skipped=use_skipped)
            _write_dataset_ids(eval_dataset, root_path, is_train=False, use_skipped=use_skipped)
        else:
            if _check_ids_match(train_dataset, root_path, eval_dataset, use_skipped=use_skipped):
                logger.info(f"Passed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
            else:
                logger.info(f"Failed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
    
    if not use_label_smoothing and not add_unsure:
        train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != "Negligibly better/unsure")
        eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != "Negligibly better/unsure")

    if preference_type != "all": # ["all", "much_better", "better", "slightly_better", "all_better"]
        if preference_type != "all_better":
            if "_" in preference_type:
                preference_type = preference_type.replace("_", " ")
            # capitalize first
            preference_type = preference_type[0].upper() + preference_type[1:]
            train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == preference_type)
            eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == preference_type)
        else:
            train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] in ["Much better", "Better"])
            eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] in ["Much better", "Better"])
    # randomly take a percentage of the dataset
    train_dataset = Dataset.from_pandas(train_dataset.shuffle(seed).to_pandas().sample(frac=percentage_used))
    eval_dataset = Dataset.from_pandas(eval_dataset.shuffle(seed).to_pandas().sample(frac=percentage_used))
    
    preprocess_function_maybe_margin = partial(
        _preprocess_dpo, 
        tokenizer=tokenizer, 
        codebook_size=codebook_size, 
        use_label_smoothing=use_label_smoothing,
        base_dir=root_path
        
    )
    train_dataset = train_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=1,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=1,
    )
    
    return train_dataset, eval_dataset
    