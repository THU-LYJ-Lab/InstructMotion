#!/bin/bash
### hyperparameters ###
BETA=0.1
USE_LABEL_SMOOTHING=false
LR_SCHEDULER_TYPE="cosine"
OPTIMIZER_TYPE="paged_adamw_32bit"
LEARNING_RATE=0.001
PER_DEVICE_TRAIN_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
EPOCHS=20
BF16=true
MAX_LENGTH=128
MAX_PROMPT_LENGTH=128
MAX_TARGET_LENGTH=128
LABEL_PAD_TOKEN_ID=-100
MAX_STEPS=1000
USE_PEFT=true
PEFT_LORA_R=8
PEFT_LORA_ALPHA=16
PEFT_LORA_DROPOUT=0.05
SANITY_CHECK=true
REPORT_TO="wandb"
IS_ENCODER_DECODER=true
ROOT_PATH="/data3/lyh/instructmotion"
OUTPUT_DIR="/data3/lyh/instructmotion/outputs"
SEED=2222
NEFTUNE_NOISE_ALPHA=0.0
LOSS_TYPE="ipo"
PERCENT_DATA=1.0
PREFERENCE_TYPE="all"
ADD_UNSURE=true
EVALUATE=true
VISUALIZE=false


# wandb
WANDB_PROJECT="motiongpt"
export WANDB_API_KEY=""
export WANDB_NAME="dpo/baseline/PEFT$PEFT,BATCH_SIZE$PER_DEVICE_TRAIN_BATCH_SIZE,LR$LEARNING_RATE,BETA$BETA,USE_LABEL_SMOOTHING$USE_LABEL_SMOOTHING,DROPOUT$DROPOUT,NEFTUNE_NOISE_ALPHA$NEFTUNE_NOISE_ALPHA,LOSS_TYPE$LOSS_TYPE,PERCENT_DATA$PERCENT_DATA,PREFERENCE_TYPE$PREFERENCE_TYPE,$ADD_UNSURE"
export WANDB_TAGS="sep_critic" 
export WANDB_RUN_GROUP="dpo"

DPO_CONFIG="""--beta $BETA \
    --use_label_smoothing $USE_LABEL_SMOOTHING \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --optimizer_type $OPTIMIZER_TYPE \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --epochs $EPOCHS \
    --bf16 $BF16 \
    --max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --label_pad_token_id $LABEL_PAD_TOKEN_ID \
    --max_steps $MAX_STEPS \
    --use_peft $USE_PEFT \
    --peft_lora_r $PEFT_LORA_R \
    --peft_lora_alpha $PEFT_LORA_ALPHA \
    --peft_lora_dropout $PEFT_LORA_DROPOUT \
    --sanity_check $SANITY_CHECK \
    --report_to $REPORT_TO \
    --is_encoder_decoder $IS_ENCODER_DECODER \
    --root_path $ROOT_PATH \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --neftune_noise_alpha $NEFTUNE_NOISE_ALPHA \
    --loss_type $LOSS_TYPE \
    --percent_data $PERCENT_DATA \
    --preference_type $PREFERENCE_TYPE \
    --add_unsure $ADD_UNSURE \
    $(if [ "$EVALUATE" = "true" ]; then echo "--evaluate"; fi) \
    $(if [ "$VISUALIZE" = "true" ]; then echo "--visualize"; fi) \
"""

CMD="""
python -m src.scripts.dpo \
    $DPO_CONFIG \
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