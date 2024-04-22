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

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, PeftModelForSeq2SeqLM
from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoModelForSeq2SeqLM

from trl import is_xpu_available
from trl import is_xpu_available, set_seed
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from src.utils import motiongpt_utils
from src.trainer.dpo_trainer import MotiongptDPOTrainer

import torch.nn as nn
import os
import json
import wandb

torch.autograd.set_detect_anomaly(True)

def remove_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

class DPOCallback(TrainerCallback):
    def __init__(self, evaluate, visualize) -> None:
        super().__init__() 
        self.evaluate=evaluate
        self.visualize=visualize

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch % 5 != 0:
            return
        if not os.path.exists("{}/checkpoint-{}-{}".format(args.output_dir, int(state.epoch), state.global_step)):
            os.mkdir("{}/checkpoint-{}-{}".format(args.output_dir, int(state.epoch), state.global_step))
        checkpoint_path = "{}/checkpoint-{}-{}/policy-checkpoint-{}-{}.pt".format(args.output_dir, int(state.epoch), state.global_step, int(state.epoch), state.global_step)
        torch.save(kwargs["model"].state_dict(), checkpoint_path)

        if self.evaluate:
            # evaluate by sending commandline
            use_peft = type(kwargs["model"]) == PeftModelForSeq2SeqLM
            peft_string = "--peft" if use_peft else ""
            if use_peft:
                r = kwargs["model"].peft_config["default"].r
                lora_alpha = kwargs["model"].peft_config["default"].lora_alpha
                lora_dropout = kwargs["model"].peft_config["default"].lora_dropout

            else:
                # put placeholders
                r = 0
                lora_alpha = 0
                lora_dropout = 0.0
                
            os.chdir("MotionGPT")
            cmd = "python test.py --cfg configs/config_eval_during_training.yaml --task t2m --checkpoint {} {} --r {} --lora_alpha {} --lora_dropout {}".format(checkpoint_path, peft_string, r, lora_alpha, lora_dropout)
            os.system(cmd)
            os.chdir("..")

            # read metrics
            metrics_path = "{}/checkpoint-{}-{}/policy-checkpoint-{}-{}-temp-1.0-metrics.json".format(args.output_dir, int(state.epoch), state.global_step, int(state.epoch), state.global_step)
            metrics = json.load(open(metrics_path))
            wandb.log(metrics)

        if self.visualize:
            # make visualizations
            dataset_root_path = args.output_dir.split("/outputs")[0]
            prompts_path = "{}/selected_prompts_test.txt".format(dataset_root_path)
            os.chdir("MotionGPT")
            os.system("python generate_npy.py --cfg configs/config_eval_during_training.yaml --task t2m --checkpoint {} {} --r {} --lora_alpha {} --lora_dropout {} --eval_examples {}".format(checkpoint_path, peft_string, r, lora_alpha, lora_dropout, prompts_path))
            data_dir = "{}/checkpoint-{}-{}/generated_npys".format(args.output_dir, int(state.epoch), state.global_step)
            video_dir = "{}/checkpoint-{}-{}/generated_videos".format(args.output_dir, int(state.epoch), state.global_step)
            
            
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            os.system("python generate_videos.py --data_dir {} --video_dir {} --prompts {}".format(data_dir, video_dir, prompts_path))
            os.chdir("..")


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    use_label_smoothing: bool = False
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    # weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=64, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Whether to activate fp16 mixed precision during training"}
    )
    bf16: Optional[bool] = field(
        default=True, metadata={"help": "Whether to activate bf16 mixed precision during training"}
    )
    max_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    # lora parameters
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"}) # originally 8
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}) # originally 16
    peft_lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the dropout parameter of the LoRA adapters"}) # originally 0.05
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    is_encoder_decoder: bool = True
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    generate_during_eval: Optional[bool] = field(default=False, metadata={"help": "Generate during evaluation"})
    seed: Optional[int] = field(default=2222, metadata={"help": "the seed for reproducibility"})

    root_path: Optional[str] = field(default=".", metadata={"help": "the root path for the dataset"})
    output_dir: Optional[str] = field(default="./outputs", metadata={"help": "the output directory"})
    dropout: Optional[bool] = field(default=False, metadata={"help": "if true, then keep the dropout in the model"})
    epochs: Optional[int] = field(default=20, metadata={"help": "the number of training epochs"})
    neftune_noise_alpha: Optional[float] = field(default=0.0, metadata={"help": "the noise alpha for neftune noise"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type for DPO loss"}) # ["sigmoid", "hinge", "ipo", "kto_pair"]
    percent_data: Optional[float] = field(default=1.0, metadata={"help": "the percentage of data to use for training"})
    preference_type: Optional[str] = field(default="all", metadata={"help": "the preference type to use for training"}) # ["all", "much_better", "better", "slightly_better"]
    add_unsure: Optional[bool] = field(default=False, metadata={"help": "add unsure samples to the training data"})
    evaluate: Optional[bool] = field(default=False, metadata={"help": "evaluate the model during training"})
    visualize: Optional[bool] = field(default=False, metadata={"help": "visualize the model during training"})


def main(script_args):
    ########## set up arguments ##########
    
    set_seed(script_args.seed)
    
    run_name = "dpo_peft{}_bs{}_lr{}_epochs{}_labelsmooth{}_r{}_alpha{}_loradropout{}_beta{}_dropout{}_neftune{}_losstype{}_data{}_preferencetype{}_addunsure{}".format(
        script_args.use_peft,
        script_args.per_device_train_batch_size,
        script_args.learning_rate,
        script_args.epochs,
        script_args.use_label_smoothing,
        script_args.peft_lora_r,
        script_args.peft_lora_alpha,
        script_args.peft_lora_dropout,
        script_args.beta,
        script_args.dropout,
        script_args.neftune_noise_alpha,
        script_args.loss_type,
        script_args.percent_data,
        script_args.preference_type,
        script_args.add_unsure,
    )
    script_args.jsonl_path="{}/preference_data/preference_labels.jsonl".format(script_args.root_path)
    script_args.data_root_path = "{}/preference_data".format(script_args.root_path)
    script_args.pretrain_ckpt_path = "{}/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar".format(script_args.root_path)
    script_args.model_name = "{}/MotionGPT/deps/flan-t5-base".format(script_args.root_path)
    
    os.environ["WANDB_NAME"] = run_name
    script_args.output_dir = os.path.join(script_args.output_dir, run_name)
    # save config
    os.makedirs(script_args.output_dir, exist_ok=True)
    with open(os.path.join(script_args.output_dir, "config.json"), "w") as f:
        json.dump(asdict(script_args), f)

    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{PartialState().local_process_index}"}
            if is_xpu_available()
            else {"": PartialState().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    ########## build tokenizer + models ##########
    
    tokenizer = motiongpt_utils.get_motion_tokenizer(tokenizer_name=script_args.model_name, codebook_size=512)
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)

    model.resize_token_embeddings(len(tokenizer))
    if not script_args.dropout:
        remove_dropout(model)
    if not script_args.use_peft:
        motiongpt_utils.load_pretrained_reward_dpo(model, script_args.pretrain_ckpt_path)
    else:
        motiongpt_utils.load_pretrained_reward_ppo_peft(model, script_args.pretrain_ckpt_path)
        
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    if not script_args.use_peft:
        model_ref = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)
        model_ref.resize_token_embeddings(len(tokenizer))
        motiongpt_utils.load_pretrained_reward_ppo(model_ref, script_args.pretrain_ckpt_path)
    else:
        # If one uses PEFT, there is no need to load a reference model
        model_ref = None

    ########## build datasets ##########
    
    train_dataset, eval_dataset = motiongpt_utils.build_dpo_dataset(
        use_label_smoothing=script_args.use_label_smoothing,
        jsonl_path=script_args.jsonl_path,
        seed=script_args.seed,
        root_path=script_args.data_root_path,
        tokenizer=tokenizer,
        codebook_size=512,
        percentage_used=script_args.percent_data,
        preference_type= script_args.preference_type, #[all, much better, better, slightly better]
        add_unsure=script_args.add_unsure,
    )
    train_dataset = {
        "prompt": train_dataset["prompt"],
        "chosen": train_dataset["chosen"],
        "rejected": train_dataset["rejected"],
        "label_smoothing": train_dataset["label_smoothing"]
    }
    eval_dataset = {
        "prompt": eval_dataset["prompt"],
        "chosen": eval_dataset["chosen"],
        "rejected": eval_dataset["rejected"],
        "label_smoothing": eval_dataset["label_smoothing"]
    }

    train_dataset = Dataset.from_dict(train_dataset)
    eval_dataset = Dataset.from_dict(eval_dataset)

    ########## set up DPO trainer ##########
    
    training_args = TrainingArguments(
        lr_scheduler_type=script_args.lr_scheduler_type,
        optim=script_args.optimizer_type,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.epochs,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="epoch",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        output_dir=script_args.output_dir,
        warmup_steps=150,
        report_to=script_args.report_to,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        save_strategy="epoch",
        gradient_checkpointing=script_args.gradient_checkpointing,
        neftune_noise_alpha=script_args.neftune_noise_alpha,
        seed=script_args.seed,
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
        
    )

    # save training arguments
    with open(os.path.join(script_args.output_dir, "training_args.json"), "w") as f:
        json.dump(asdict(training_args), f)

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=script_args.peft_lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",

        )
    else:
        peft_config = None

    # initialize the DPO trainer
    dpo_trainer = MotiongptDPOTrainer(
        model,
        model_ref,
        use_label_smoothing=script_args.use_label_smoothing,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=script_args.generate_during_eval,
        peft_config=peft_config,
        is_encoder_decoder=script_args.is_encoder_decoder,
        callbacks=[DPOCallback(evaluate=script_args.evaluate, visualize=script_args.visualize)],
        loss_type=script_args.loss_type, # ["sigmoid", "hinge", "ipo", "kto_pair"]
    )

    ######### start training ########## 
    
    dpo_trainer.train()

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
