# Official repository for CVPRW 2024 HuMoGen paper "Exploring Text-to-Motion Generation with Human Preference"

## Directory structure
**`assets/`**: contains the generated motions from MotionGPT, PPO, and DPO models for the demo.

**`checkpoints/`**: 
- `MotionGPT-base/` - finetuned MotionGPT model.
- `dpo/` - our finetuned DPO model
- `ppo/` - our finetuned RLHF PPO model
- `rm` - our finetuned reward model

**`commands/`**:
- `ppo_train.sh` - script for training PPO model with shared base model and separate value and policy heads
- `ppo_sep_critic_train.sh` - script for training PPO model with separate value and policy models
- `dpo_train.sh` - script for training DPO model
- `rm_train` - script for training reward model

**`preference_data/`**: preference dataset

**`MotionGPT/`**: the [MotionGPT codebase](https://github.com/OpenMotionLab/MotionGPT) with the following changes
- We changed the following files
  - MotionGPT/test.py
  - MotionGPT/mGPT/config.py
  - MotionGPT/mGPT/utils/load_checkpoint.py
  - MotionGPT/requirements.txt
  - MotionGPT/mGPT/archs/mgpt_lm.py

- We added the following files
  - MotionGPT/configs/config_eval_during_training.yaml
  - MotionGPT/generate_npy.py
  - MotionGPT/generate_videos.py

**`src/`**:
- `models/` - scripts for training and evaluation
- `scripts/` - scripts for running experiments
- `trainer/` - scripts for training and evaluation



## Installations
1. Download the preference dataset at [cloud (code xky8)](https://pan.baidu.com/s/1-07tLTZdEdsHfMfBHMctuA) and put it at `preference_data/`
3. Download our checkpoints from [cloud (code sxx5)](https://pan.baidu.com/s/1euBCQOE2EG90VHj94Yty-Q) and put them at `checkpoints/`
2. Download the HumanML3D dataset from [https://github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D), preprocess it according to their instructions, and put it under `MotionGPT/datasets/`
3. Set up environment according to [MotionGPT](https://github.com/OpenMotionLab/MotionGPT?tab=readme-ov-file#-quick-start) setup instructions below:
```conda create python=3.10.6 --name mgpt
conda activate mgpt
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
cd MotionGPT
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash prepare/download_smpl_model.sh
bash prepare/prepare_t5.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_pretrained_models.sh
```

5. (optional) set up visualization dependencies. Please refer to [MotionGPT](https://github.com/OpenMotionLab/MotionGPT?tab=readme-ov-file#-visualization) for set up instructions.

## Training
To train the reward model, modify the hyperparameters and paths in `src/scripts/rm_train.sh` and run the following command:
```
bash src/scripts/rm_train.sh
```

To train the PPO model with shared base model and separate value and policy heads, modify the hyperparameters and paths in `src/scripts/ppo_train.sh` and run the following command (does not support PEFT):
```
bash src/scripts/ppo_train.sh
```

To train the PPO model with separate value and policy models, modify the hyperparameters and paths in `src/scripts/ppo_sep_critic_train.sh` and run the following command (does not support PEFT):
```
bash src/scripts/ppo_sep_critic_train.sh
```

To train the DPO model, modify the hyperparameters and paths in `src/scripts/dpo_train.sh` and run the following command (does not support distributed training):
```
bash src/scripts/dpo_train.sh
```

## Evaluation
```cd``` into MotionGPT first.

To evaluate without peft:
```
python test.py --cfg configs/config_h3d_stage3.yaml --task t2m --checkpoint /path/to/trained_model.pt
```

To evaluate with peft:
```
python test.py --cfg configs/config_h3d_stage3.yaml --task t2m --checkpoint /path/to/trained_model.pt --peft --r 8 --lora_alpha 16 --lora_dropout 0.05 
```

## Visualization
To generate npy files for visualization:
```
python generate_npy.py --cfg configs/config_h3d_stage3.yaml --task t2m --checkpoint /path/to/trained_model.pt --peft --r 8 --lora_alpha 16 --lora_dropout 0.05 
```

To generate videos for visualization:
```
python generate_videos.py --data_dir /path/to/generated_npys --video_dir /path/to/generated_videos
```

## Demo
We provide a demo of the generated motions from MotionGPT, PPO, and DPO models (with temperature 1.0). The following table shows the generated motions for the given text instructions.
| Text Instruction | MotionGPT Generated Motion | PPO Generated Motion | DPO Generated Motion |
| --- | --- | --- | --- |
| "the individual is shaking their head from side to side" |  <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/61183785-282b-4878-8f99-cc8704bf3e61" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/bf5b6f3f-ea9e-45f7-b6ba-6510d41a2cd8" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/569fe37e-a2d7-40da-a8f7-9dc822dda167" />  |
| "someone leaps off a concrete block" | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/35e7fa70-39f4-466a-a603-dc82a0bbe0eb" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/9806ab90-8925-4189-a129-55ff08ad2736" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/bda5975e-5dcc-4591-a386-081cc9669282" />  |
| "a person lifts their arms, widens the space between their legs, and joins their hands together" | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/912058ca-b139-4325-b4b3-a658236a0c9f" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/a35767f0-e5c0-4870-b9f7-cbaa4e22c395" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/870ba389-89e0-435f-941e-d9d30365381b" />  |
| "he moves his feet back and forth while dancing.." |  <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/3ee80a14-292f-4b7b-b38c-e08b3bda0a34" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/312a7429-85ee-4a99-a5ea-16558981e5ef" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/c9cb0d9f-5a32-42be-bea3-5874e64e0744" />  |
| "move the body vigorously and then plop down on the ground" | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/563fd1bd-5940-4880-a387-289284a44c1e" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/268b4eb8-600f-4608-b0e7-8a9ac8932e33" /> | <video src="https://github.com/THU-LYJ-Lab/InstructMotion/assets/60765890/55dc797a-9c35-491a-8f88-4a9fb31b1934" />  |

## Citation
If you find this code useful, please consider citing our paper:
```
@misc{sheng2024exploring,
      title={Exploring Text-to-Motion Generation with Human Preference}, 
      author={Jenny Sheng and Matthieu Lin and Andrew Zhao and Kevin Pruvost and Yu-Hui Wen and Yangguang Li and Gao Huang and Yong-Jin Liu},
      year={2024},
      eprint={2404.09445},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Acknowledgments
Thank you to MotionGPT authors for providing the [codebase](https://github.com/OpenMotionLab/MotionGPT) and the [finetuned model](https://huggingface.co/OpenMotionLab/MotionGPT-base). Our code is partially borrowing from them.

## License
This code is distributed under an MIT LICENSE.

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
