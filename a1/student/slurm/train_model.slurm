#!/bin/bash

#SBATCH --job-name=ab12057_csciga3033_llm_reasoners_a1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a1
#SBATCH --mail-type=all
#SBATCH --mail-user=ab12057@nyu.edu

#interactive
#n2c48m24 → CPU only
#g2-standard-12 → 1 × L4 GPU
#g2-standard-24 → 2 × L4 GPUs
#g2-standard-48 → 4 × L4 GPUs
#c12m85-a100-1 → 1 × A100 40GB GPU
#c24m170-a100-2 → 2 × A100 40GB GPUs

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source .venv/bin/activate

# Data configuration
TRAIN_DATA="data/TinyStoriesV2-GPT4-train-10k.npy"
VAL_DATA="data/TinyStoriesV2-GPT4-valid-10k.npy"
TOKENIZER_PATH="tokenizer/tokenizer_10k"

# Model architecture
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
ROPE_THETA=10000.0
USE_ROPE=true
EPS=1e-5
USE_NORM=true
NORM_POSITION="pre"
FFN_TYPE="swiglu"
TIED_WEIGHTS=true

# Training hyperparameters
TOTAL_STEPS=20000
BATCH_SIZE=64
EVAL_BATCHES=100
LEARNING_RATE=2e-4
MIN_LR=1e-6
WARMUP_RATIO=0.1

# Optimizer
WEIGHT_DECAY=0.1
BETA1=0.9
BETA2=0.95
OPT_EPS=1e-8
GRAD_CLIP=1.0

# Logging and checkpointing
LOG_INTERVAL=10
EVAL_INTERVAL=200
SAVE_INTERVAL=2000
OUTPUT_DIR="./saved_models/baseline_10k"
SAVE_BEST=true
SAVE_LATEST=true
CHECKPOINT=""

# Weights & Biases
USE_WANDB=true
WANDB_PROJECT="transformer-lm-training"
WANDB_ENTITY="csciga3033-llm-reasoners"
WANDB_NAME="baseline-10k"

# Text generation
GENERATE_SAMPLES=true
PROMPT="Once upon a time"
MAX_NEW_TOKENS=50
TOP_P=0.9
TEMPERATURE=0.8

# System
DEVICE="cuda"
SEED=42

# Start building the command
CMD="uv run student/train_model.py"

# Add Data Arguments
CMD="$CMD --train-data $TRAIN_DATA --val-data $VAL_DATA --tokenizer-path $TOKENIZER_PATH"

# Add Model Architecture Arguments
CMD="$CMD --context-length $CONTEXT_LENGTH --d-model $D_MODEL --num-layers $NUM_LAYERS --num-heads $NUM_HEADS --d-ff $D_FF --rope-theta $ROPE_THETA --norm-position $NORM_POSITION --ffn-type $FFN_TYPE --eps $EPS"
if [ "$USE_ROPE" = true ]; then CMD="$CMD --use-rope"; fi
if [ "$USE_NORM" = true ]; then CMD="$CMD --use-norm"; fi
if [ "$TIED_WEIGHTS" = true ]; then CMD="$CMD --tied-weights"; fi

# Add Training/Optimizer Arguments
CMD="$CMD --total-steps $TOTAL_STEPS --batch-size $BATCH_SIZE --eval-batches $EVAL_BATCHES --learning-rate $LEARNING_RATE --min-learning-rate $MIN_LR --warmup-ratio $WARMUP_RATIO --grad-clip $GRAD_CLIP"
CMD="$CMD --weight-decay $WEIGHT_DECAY --beta1 $BETA1 --beta2 $BETA2 --optimizer-eps $OPT_EPS"

# Add Logging/Checkpointing
CMD="$CMD --log-interval $LOG_INTERVAL --eval-interval $EVAL_INTERVAL --save-interval $SAVE_INTERVAL --output-dir $OUTPUT_DIR"
if [ "$SAVE_BEST" = true ]; then CMD="$CMD --save-best"; fi
if [ "$SAVE_LATEST" = true ]; then CMD="$CMD --save-latest"; fi
if [ -n "$CHECKPOINT" ]; then CMD="$CMD --checkpoint $CHECKPOINT"; fi

# Add W&B
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --wandb --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY"
    if [ -n "$WANDB_NAME" ]; then CMD="$CMD --wandb-name $WANDB_NAME"; fi
fi

# Add Generation
if [ "$GENERATE_SAMPLES" = true ]; then
    CMD="$CMD --generate-samples --prompt \"$PROMPT\" --max-new-tokens $MAX_NEW_TOKENS --top-p $TOP_P --temperature $TEMPERATURE"
fi

# Add System
CMD="$CMD --device $DEVICE --seed $SEED"

# Delete previous model checkpoints if OUTPUT_DIR exists
if [ -d "$OUTPUT_DIR" ] && [ -z "$CHECKPOINT" ]; then
    echo "Clearing old checkpoints in $OUTPUT_DIR"
    rm -rf "${OUTPUT_DIR:?}/"*
fi

# Run the training script
eval $CMD

# Deactivate the environment
deactivate

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"