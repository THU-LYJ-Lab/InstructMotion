# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List, Dict

import json
import torch
import tyro
import transformers
from accelerate import Accelerator
from tqdm import tqdm
from transformers import T5ForSequenceClassification
from trl import is_xpu_available, set_seed
import wandb
import os 
from src.utils import motiongpt_utils

from trl import ( 
    AutoModelForSeq2SeqLMWithValueHead, 
    set_seed
)
from src.trainer.ppo_config import PPOConfig
from src.trainer.ppo_trainer import MotiongptPPOTrainer
from src.utils import motiongpt_utils
from trl.import_utils import is_npu_available, is_xpu_available


tqdm.pandas()


@dataclass
class ScriptArguments:
    
    debug_mode: bool = False
    """when debug mode is on, all saves are disabled"""
    
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="./MotionGPT/deps/flan-t5-base",
            learning_rate=1e-5,
            log_with="wandb",
            mini_batch_size=16,
            batch_size=16,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="motion", # choose from "kl", "abs", "mse", "full", "motion"
            seed=42,
            use_score_scaling=False, # this needs to be toggled
            use_score_norm=False,
            score_clip=None,
            ppo_epochs=4, # this is the inner loop of the PPO algorithm, and 4 is the default value
            optimize_device_cache=True,

        )
    )
    root_path: str = "."
    """path to the project"""
    reward_model_path: str = "checkpoints/rm"
    """path to the reward model checkpoint"""
    epochs: int = 20
    """number of epochs to train the PPO algorithm"""
    output_dir: str = "outputs"
    """output directory for the PPO algorithm"""
    max_length: int = 60
    """maximum length of the generated text"""
    min_length: int = -1
    """minimum length of the generated text"""
    temperature: float = 1.0
    """temperature for sampling"""
    evaluate: bool = False
    """evaluate the model during training"""
    visualize: bool = False
    """visualize the model during training"""
    
    
def pipeline_rm(
    batch: Dict[List[str], List[str]],
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizerBase,
    reward_model: T5ForSequenceClassification
) -> List[torch.Tensor]:
    """
    Processes a batch of texts for evaluation using a given tokenizer and reward model.

    Args:
        batch (List[str]): Batch of texts to be evaluated.
        device (torch.device): The device to perform computation on.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer for text processing.
        reward_model (transformers.T5ForSequenceClassification): Model to evaluate the texts.

    Returns:
        List[torch.Tensor]: The logits as computed by the reward model.
    """
    ENDS_WITH_EOS_TOKEN = lambda r: r[-4:] == tokenizer.eos_token
    # we're adding this because motiongpt also has two eos_tokens at the end, one here and one from the tokenizer
    texts = [q + r if ENDS_WITH_EOS_TOKEN(r) else q + r + tokenizer.eos_token for q, r in zip(batch["query"], batch["response"])]
    texts_tokenized = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=256,
    )
    rewards= reward_model(texts_tokenized["input_ids"].to(device)).logits
    return list(rewards)


def evaluate_during_training(ppo_trainer, model, run_name, epoch, step_cnt, args):
    # save model
    checkpoint_path_root = "{}/{}/checkpoint-{}-{}".format(args.output_dir, run_name, epoch, step_cnt)
    ppo_trainer.accelerator.save_state(checkpoint_path_root)
    checkpoint_path = "{}/checkpoint-{}-{}.pt".format(checkpoint_path_root, epoch, step_cnt)
    torch.save(ppo_trainer.accelerator.unwrap_model(model).pretrained_model.state_dict(), checkpoint_path)

    if args.evaluate:
        # evaluate by sending commandline
        os.chdir("MotionGPT")
        os.system("python test.py --cfg configs/config_eval_during_training.yaml --task t2m --checkpoint {}".format(checkpoint_path))
        os.chdir("..")

        # read metrics
        metrics_path = "{}/checkpoint-{}-{}-temp-{}-metrics.json".format(checkpoint_path_root, epoch, step_cnt, args.temperature)
        metrics = json.load(open(metrics_path))
        wandb.log(metrics)
    
    if args.visualize:
        # make visualizations
        prompts_path = "{}/selected_prompts_test.txt".format(args.data_root_path)
        os.chdir("MotionGPT")
        os.system("python generate_npy.py --cfg configs/config_eval_during_training.yaml --task t2m --checkpoint {} --eval_examples {}".format(checkpoint_path, prompts_path))
        data_dir = "{}/generated_npys_temp_{}".format(checkpoint_path_root, args.temperature)
        video_dir = "{}/generated_videos_temp_{}".format(checkpoint_path_root, args.temperature)
        os.system("python generate_videos.py --data_dir {} --video_dir {} --prompts {}".format(data_dir, video_dir, prompts_path))
        os.chdir("..")
                
def main(args):
    ########## set up arguments ##########
    
    run_name = "ppo_bs{}_lr{}_epochs{}_kl{}".format(
        args.ppo_config.batch_size,
        args.ppo_config.learning_rate,
        args.epochs,
        args.ppo_config.kl_penalty,
    )
    args.ppo_config.project_kwargs = {"project_dir": "{}/{}".format(args.output_dir, run_name)}
    args.jsonl_path="{}/preference_data/preference_labels.jsonl".format(args.root_path)
    args.data_root_path = "{}/preference_data".format(args.root_path)
    args.pretrain_ckpt_path = "{}/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar".format(args.root_path)

    ########## build tokenizer + datasets ##########

    tokenizer = motiongpt_utils.get_motion_tokenizer(tokenizer_name=args.ppo_config.model_name, codebook_size=512)
    train_dataset, eval_dataset = motiongpt_utils.build_ppo_dataset(
        jsonl_path=args.jsonl_path,
        seed=args.ppo_config.seed,
        root_path=args.data_root_path,
        tokenizer=tokenizer
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ########## set up models ##########
    
    # set seed before initializing value head
    set_seed(args.ppo_config.seed)
    trl_model_class = AutoModelForSeq2SeqLMWithValueHead
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name)
    ref_model.pretrained_model.resize_token_embeddings(len(tokenizer))
    motiongpt_utils.load_pretrained_reward_ppo(ref_model, args.pretrain_ckpt_path)

    model = trl_model_class.from_pretrained(args.ppo_config.model_name)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    motiongpt_utils.load_pretrained_reward_ppo(model, args.pretrain_ckpt_path)

    ########## set up PPO trainer ##########
    
    ppo_trainer = MotiongptPPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=train_dataset, data_collator=collator)
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": args.min_length,
        "max_length": args.max_length,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # "max_new_tokens": 32,
    }

    ########## set up reward model ##########
    
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            reward_model = T5ForSequenceClassification.from_pretrained(args.reward_model_path, num_labels=1).to(device)
    else:
        reward_model = T5ForSequenceClassification.from_pretrained(args.reward_model_path, num_labels=1).to(device)
    
    ########## save pretrained model to make sure that save works, this is important if using peft ##########
    
    step_cnt = 1
    if ppo_trainer.accelerator.is_main_process and not args.debug_mode:
        epoch = "pretrained"
        evaluate_during_training(ppo_trainer, model, run_name, epoch, step_cnt, args)
                
    
    ######### start training ########## 
    progress_bar = tqdm(range(args.epochs * len(ppo_trainer.dataloader)), disable = not ppo_trainer.accelerator.is_main_process)
    progress_bar.set_description("Step")
    for epoch in range(args.epochs):
        for _, batch in enumerate(ppo_trainer.dataloader):
            
            query_tensors = batch["input_ids"]
            query_tensors = list(torch.LongTensor(query_tensors).to(device).squeeze(1))

            # Get response
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors, 
                return_prompt=False, 
                batch_size=1, 
                generate_ref_response=True,
                **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

            # Compute reward
            rewards = pipeline_rm(batch, device, tokenizer, reward_model)
            batch["rewards"] = rewards
            ref_rewards = pipeline_rm(batch, device, tokenizer, reward_model)
            batch["ref_rewards"] = ref_rewards

            # Run PPO step
            stats = ppo_trainer.step(
                query_tensors, 
                response_tensors, 
                rewards
            )
            
            ppo_trainer.log_stats(
                stats, 
                batch, 
                rewards, 
                ref_rewards, 
                columns_to_log=["query", "response", "ref_response", "ref_rewards", "rewards"]
            )

            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "reward": float(sum(rewards)/len(rewards)),
                    "ref_reward": float(sum(ref_rewards)/len(ref_rewards))
                }
            )
            
            step_cnt += 1

            # # if we reached half way through one epoch, save model and evaluate
            # if (ppo_trainer.accelerator.is_main_process and not args.debug_mode) and step_cnt == len(ppo_trainer.dataloader)//2 and step_cnt != 0:
            #     evaluate_during_training(ppo_trainer, model, run_name, epoch, step_cnt, args)

        if ppo_trainer.accelerator.is_main_process and not args.debug_mode and epoch % 5 == 0:
            evaluate_during_training(ppo_trainer, model, run_name, epoch, step_cnt, args)


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)
    main(args)