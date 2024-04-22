SAVE_DIR="/data3/lyh/instructmotion/outputs"
ROOT_PATH="/data3/lyh/instructmotion"

# hyperparameters
SEED=42
USE_MARGIN=true
MARGIN_TYPE="big"
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=20
WEIGHT_DECAY=0.0
NEFTUNE_NOISE_ALPHA=0.1
BATCH_SIZE=32
USE_UNSURE=false
LOGGING_FIRST_STEP=true
REPORT_TO="wandb"
LOGGING_STEPS=20
EVALUATION_STRATEGY="epoch"
DISABLE_TQDM=true
MAX_LENGTH=256
OPTIM="adamw_torch"
ADAM_BETA1=0.9
ADAM_BETA2=0.999
GRADIENT_CHECKPOINTING=false
USE_REENTRANT=false
REMOVE_UNUSED_COLUMNS=false
DDP_FIND_UNUSED_PARAMETERS=false



# wandb
export WANDB_PROJECT="motiongpt"
export WANDB_API_KEY=""
export WANDB_NAME="rm/baseline"
export WANDB_TAGS="big_margin, neftune"

# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS="" # --mixed_precision 'fp16'

# Set your number of GPUs here
IS_DEBUG=$1
if (( IS_DEBUG == 1 )); then
    SAVE_STRATEGY="no"
    NUM_GPUS=1
    export WANDB_MODE=disabled
else
    SAVE_STRATEGY="epoch"
    NUM_GPUS=1
fi

REWARD_CONFIG="""--learning_rate $LEARNING_RATE \
    --output_dir $SAVE_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --optim "adamw_torch" \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay $WEIGHT_DECAY \
    --neftune_noise_alpha $NEFTUNE_NOISE_ALPHA \
    --save_strategy $SAVE_STRATEGY \
    --seed $SEED \
    --per_device_train_batch_size $BATCH_SIZE \
    $(if [ "$LOGGING_FIRST_STEP" = "true" ]; then echo "--logging_first_step"; fi) \
    --logging_steps $LOGGING_STEPS \
    --evaluation_strategy $EVALUATION_STRATEGY \
    $(if [ "$DISABLE_TQDM" = "true" ]; then echo "--disable_tqdm True"; fi) \
    --max_length $MAX_LENGTH \
    $(if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then echo "--gradient_checkpointing"; fi) \
    $(if [ "$USE_REENTRANT" = "true" ]; then echo "--gradient_checkpointing_kwargs.reentrant True"; fi) \
    $(if [ "$REMOVE_UNUSED_COLUMNS" = "true" ]; then echo "--remove_unused_columns"; fi) \
    $(if [ "$DDP_FIND_UNUSED_PARAMETERS" = "true" ]; then echo "--ddp_find_unused_parameters"; fi) \
    --report_to $REPORT_TO \
"""

CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --num_processes $NUM_GPUS \
    --main_process_port 29502 \
    -m src.scripts.reward_modeling \
    $(if [ "$USE_MARGIN" = "true" ]; then echo "--use_margin"; fi) \
    --margin_type $MARGIN_TYPE \
    --root_path $ROOT_PATH \
    $(if [ "$USE_UNSURE" = "true" ]; then echo "--use_unsure"; fi) \
    $REWARD_CONFIG
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