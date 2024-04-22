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
import logging
from dataclasses import dataclass, field
from functools import partial
import os

import wandb
import tyro
import numpy as np
import transformers
from tqdm import tqdm
from transformers import T5ForSequenceClassification, HfArgumentParser
from trl import set_seed
from transformers import logging as transformers_logging
from transformers.integrations import WandbCallback


from src.trainer.reward_trainer import RewardTrainer
from src.utils import motiongpt_utils
from src.trainer.reward_config import RewardConfig

tqdm.pandas()


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
    """

    def __init__(
        self, 
        trainer,
        val_dataset,
        train_dataset
    ):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
        """
        super().__init__()
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset.select(range(len(val_dataset)))

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            train_predictions = self.trainer.predict(self.train_dataset)
            train_logits = train_predictions.predictions
            train_chosen_logits = train_logits[:, 0]
            train_rejected_logits = train_logits[:, 1]
            self._wandb.log({"train/chosen_reward_distribution": wandb.Histogram(np_histogram=np.histogram(train_chosen_logits))})
            self._wandb.log({"train/rejected_reward_distribution": wandb.Histogram(np_histogram=np.histogram(train_rejected_logits))})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            val_predictions = self.trainer.predict(self.val_dataset)
            val_logits = val_predictions.predictions
            val_chosen_logits = val_logits[:, 0]
            val_rejected_logits = val_logits[:, 1]
            self._wandb.log({"eval/chosen_reward_distribution": wandb.Histogram(np_histogram=np.histogram(val_chosen_logits))})
            self._wandb.log({"eval/rejected_reward_distribution": wandb.Histogram(np_histogram=np.histogram(val_rejected_logits))})


@dataclass
class ScriptArguments:
    root_path: str = field(default=".", metadata={"help": "path to the project"})
    use_margin: bool = field(default=False, metadata={"help": "whether to use margin loss"})
    margin_type: str = field(default="big", metadata={"help": "big margin puts more emphasis on much better"})
    sweep: bool = field(default=False, metadata={"help": "whether to use optuna hyperparameter sweep"})
    use_unsure: bool = field(default=False, metadata={"help": "whether to use unsure samples"})
    use_skipped: bool = field(default=False, metadata={"help": "whether to use skipped samples"})
    
    

def model_init(
    tokenizer: transformers.PreTrainedTokenizer,
    model_name: str, 
    pretrain_ckpt_path: str,
):
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    motiongpt_utils.load_pretrained_reward_rm(model, pretrain_ckpt_path)
    return model

def main(args):
    args.model_name = os.path.join(args.root_path, "MotionGPT/deps/flan-t5-base")
    args.pretrain_ckpt_path = os.path.join(args.root_path, "checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar")
    args.dataset_path = os.path.join(args.root_path, "preference_data/preference_labels.jsonl")
    args.data_root_path = "{}/preference_data".format(args.root_path)
    name_with_config = "/epoch{}_seed{}_bs{}_lr{}_margin{}_neftune{}_wd{}_unsure{}_skipped{}".format(
        args.reward_config.num_train_epochs,
        args.reward_config.seed,
        args.reward_config.per_device_train_batch_size * args.reward_config.world_size,
        args.reward_config.learning_rate,
        str(args.use_margin)+args.margin_type,
        args.reward_config.neftune_noise_alpha,
        args.reward_config.weight_decay,
        args.use_unsure,
        args.use_skipped
    )
    args.reward_config.output_dir = args.reward_config.output_dir + name_with_config
    
    colorful_handler = motiongpt_utils.get_colorful_handler()
    transformers_logging.set_verbosity(logging.INFO)
    transformers_logging.disable_default_handler()
    transformers_logging.add_handler(colorful_handler)
    logger = transformers_logging.get_logger("transformers")

    set_seed(args.reward_config.seed)
    
    # Step 1: Load the model
    tokenizer = motiongpt_utils.get_motion_tokenizer(tokenizer_name=args.model_name, codebook_size=512)
    model_init_fn = partial(model_init, tokenizer, args.model_name, args.pretrain_ckpt_path)
    model = model_init_fn()

    # Step 2: Load the dataset and pre-process it    
    train_dataset, eval_dataset = motiongpt_utils.build_rm_dataset(
        jsonl_path=args.dataset_path,
        root_path=args.data_root_path,
        tokenizer=tokenizer,
        codebook_size=512,
        use_margin=args.use_margin,
        margin_type=args.margin_type,
        use_skipped=args.use_skipped
    )
    
    if not args.use_unsure:
        train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != 'Negligibly better/unsure')
    logger.info("Filtering for evaluation dataset")
    eval_dataset = {
        'all': eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != 'Negligibly better/unsure'),
        'much_better': eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == 'Much better'),
        'better': eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == 'Better'),
        'slightly_better': eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == 'Slightly better'),
    }
    logger.info(train_dataset)
    logger.info(eval_dataset)

    # Step 5: Define the Trainer
    args.reward_config.run_name = os.environ["WANDB_NAME"]
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args.reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
        
    if args.sweep:
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 2e-5, 2e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8]),
            }
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=20,
        )
        logger.info(best_trial)
    else:
        trainer.train()
        eval_output = trainer.predict(
                test_dataset = eval_dataset['all'],
                metric_key_prefix="eval"
            )
        if trainer.state.is_world_process_zero:
            val_logits = eval_output.predictions
            val_chosen_logits = val_logits[:, 0]
            val_rejected_logits = val_logits[:, 1]
            wandb.log({"eval/chosen_reward_distribution": wandb.Histogram(np_histogram=np.histogram(val_chosen_logits))})
            wandb.log({"eval/rejected_reward_distribution": wandb.Histogram(np_histogram=np.histogram(val_rejected_logits))})
        trainer.accelerator.wait_for_everyone()
    
    wandb.finish()

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig))
    args, rm_args = parser.parse_args_into_dataclasses()
    args.reward_config = rm_args
    main(args)