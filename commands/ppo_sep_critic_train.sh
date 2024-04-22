#!/bin/bash
### hyperparameters ###
SEED=42
LEARNING_RATE=1e-05
KL_PENALTY="kl"
ADAP_KL_CTRL=true
INIT_KL_COEF=0.05
VF_COEF=0.1
BATCH_SIZE=8
MINI_BATH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
PPO_EPOCHS=4
MAX_GRAD_NORM=None
TARGET_KL=6.0
USE_SCORE_SCALING=true
USE_SCORE_NORM=true
SCORE_CLIP=None
WORLD_SIZE=4
GLOBAL_BACKWARD_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=32
EPOCHS=20
MAX_LENGTH=60
MIN_LENGTH=10
CLIPRANGE_VALUE=0.2
HORIZON=10000
TEMPERATURE=1.0
REMOVE_DROPOUT=true
WHITEN_REWARDS=true
EVALUATE=true
VISUALIZE=false

ROOT_PATH="/data3/lyh/instructmotion"
OUTPUT_DIR="/data3/lyh/instructmotion/outputs"
MODEL_NAME="/data3/lyh/instructmotion/MotionGPT/deps/flan-t5-base"
REWARD_MODEL_PATH="/data3/lyh/instructmotion/checkpoints/rm"

# Set your number of GPUs here
IS_DEBUG=$1
if (( IS_DEBUG == 1 )); then
    DEBUG_MODE="--debug_mode"
    NUM_GPUS=1
    export WANDB_MODE=disabled
else
    DEBUG_MODE=""
    NUM_GPUS=2
fi
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# wandb
WANDB_PROJECT="motiongpt"
export WANDB_API_KEY=""
export WANDB_NAME="ppo/baseline/lr_$LEARNING_RATE,kl_$KL_PENALTY,adap_kl_ctrl_$ADAP_KL_CTRL,init_kl_coef_$INIT_KL_COEF,use_score_norm_$USE_SCORE_NORM,kl_penalty_$KL_PENALTY,ppo_epochs_$PPO_EPOCHS,batch_size_$TOTAL_BATCH_SIZE"
export WANDB_TAGS="sep_critic" 
export WANDB_RUN_GROUP="ppo"

PPO_CONFIG="""--ppo_config.learning_rate $LEARNING_RATE \
    --ppo_config.mini_batch_size $MINI_BATH_SIZE \
    --ppo_config.batch_size $BATCH_SIZE \
    --ppo_config.ppo_epochs $PPO_EPOCHS \
    --ppo_config.kl_penalty $KL_PENALTY \
    $(if [ "$USE_SCORE_SCALING" = "true" ]; then echo "--ppo_config.use_score_scaling"; fi) \
    $(if [ "$USE_SCORE_NORM" = "true" ]; then echo "--ppo_config.use_score_norm"; fi) \
    $(if [ "$ADAP_KL_CTRL" = "true" ]; then echo "--ppo_config.adap_kl_ctrl"; fi) \
    $(if [ "$WHITEN_REWARDS" = "true" ]; then echo "--ppo_config.whiten_rewards"; fi) \
    --ppo_config.init_kl_coef $INIT_KL_COEF \
    --ppo_config.vf_coef $VF_COEF \
    --ppo_config.max_grad_norm $MAX_GRAD_NORM \
    --ppo_config.cliprange_value $CLIPRANGE_VALUE \
    --ppo_config.seed $SEED \
    --ppo_config.target_kl $TARGET_KL \
    --ppo_config.horizon $HORIZON \
    --ppo_config.gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ppo_config.score_clip $SCORE_CLIP \
    --ppo_config.tracker_project_name $WANDB_PROJECT \
    --ppo_config.world_size $WORLD_SIZE \
    --ppo_config.global_backward_batch_size $GLOBAL_BACKWARD_BATCH_SIZE \
    --ppo_config.global_batch_size $GLOBAL_BATCH_SIZE \
    --ppo_config.model_name $MODEL_NAME \
"""

# --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --num_processes $NUM_GPUS \
    --main_process_port 29534 \
    --mixed_precision bf16 \
    -m src.scripts.ppo_sep_critic \
    --epochs $EPOCHS \
    --max_length $MAX_LENGTH \
    --min_length $MIN_LENGTH \
    --temperature $TEMPERATURE \
    $(if [ "$REMOVE_DROPOUT" = "true" ]; then echo "--remove_dropout"; fi) \
    --root_path $ROOT_PATH \
    --output_dir $OUTPUT_DIR \
    --reward_model_path $REWARD_MODEL_PATH \
    $(if [ "$EVALUATE" = "true" ]; then echo "--evaluate"; fi) \
    $(if [ "$VISUALIZE" = "true" ]; then echo "--visualize"; fi) \
    $PPO_CONFIG \
    $DEBUG_MODE 
"""

echo "Starting program..."

{ # try
    echo $CMD
    eval "$CMD"
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0