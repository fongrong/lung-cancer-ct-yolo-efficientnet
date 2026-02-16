#!/bin/bash
# =============================================================================
# Train All Models Script
# Lung Cancer CT Classification - Hybrid YOLO-EfficientNet
# 
# This script trains all model variants across different learning rate configs.
# Total: 4 models × 3 LR configs = 12 training runs
#
# Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
# =============================================================================

set -e  # Exit on error

# Configuration
DATA_YAML="configs/data.yaml"
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
DEVICE=0
PROJECT="runs/train"
SEED=42
WORKERS=8
PATIENCE=50

# Models to train
MODELS=("yolov8m" "yolov8m_effnet" "yolov9m" "yolov9m_effnet")

# Learning rate configurations
declare -A LR_HIGH=( ["lr0"]="0.01" ["lrf"]="0.1" ["name"]="high" )
declare -A LR_MEDIUM=( ["lr0"]="0.003" ["lrf"]="0.01" ["name"]="medium" )
declare -A LR_LOW=( ["lr0"]="0.001" ["lrf"]="0.005" ["name"]="low" )

# Start time
START_TIME=$(date +%s)

echo "=============================================================="
echo "TRAINING ALL MODELS"
echo "=============================================================="
echo "Models:      ${MODELS[*]}"
echo "LR configs:  high, medium, low"
echo "Epochs:      $EPOCHS"
echo "Batch size:  $BATCH_SIZE"
echo "Image size:  $IMG_SIZE"
echo "Device:      $DEVICE"
echo "=============================================================="

# Counter for progress
TOTAL_RUNS=12
CURRENT_RUN=0

# Function to train a model
train_model() {
    local MODEL=$1
    local LR0=$2
    local LRF=$3
    local LR_NAME=$4
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    NAME="${MODEL}_${LR_NAME}_lr${LR0}"
    
    echo ""
    echo "=============================================================="
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Training: $NAME"
    echo "=============================================================="
    echo "Model:         $MODEL"
    echo "Learning rate: $LR0 -> $(echo "$LR0 * $LRF" | bc)"
    echo "=============================================================="
    
    python src/train.py \
        --model "$MODEL" \
        --data "$DATA_YAML" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --img-size $IMG_SIZE \
        --lr0 "$LR0" \
        --lrf "$LRF" \
        --device $DEVICE \
        --project "$PROJECT" \
        --name "$NAME" \
        --seed $SEED \
        --workers $WORKERS \
        --patience $PATIENCE
    
    echo "Completed: $NAME"
    echo ""
}

# Train all combinations
for MODEL in "${MODELS[@]}"; do
    # High LR
    train_model "$MODEL" "${LR_HIGH[lr0]}" "${LR_HIGH[lrf]}" "${LR_HIGH[name]}"
    
    # Medium LR (recommended)
    train_model "$MODEL" "${LR_MEDIUM[lr0]}" "${LR_MEDIUM[lrf]}" "${LR_MEDIUM[name]}"
    
    # Low LR
    train_model "$MODEL" "${LR_LOW[lr0]}" "${LR_LOW[lrf]}" "${LR_LOW[name]}"
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "=============================================================="
echo "ALL TRAINING COMPLETE"
echo "=============================================================="
echo "Total runs:  $TOTAL_RUNS"
echo "Total time:  ${HOURS}h ${MINUTES}m"
echo "Results in:  $PROJECT"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate all models:"
echo "     python src/evaluate.py --eval-all --results-dir $PROJECT"
echo ""
echo "  2. Generate comparison figures:"
echo "     python utils/visualization.py --results-dir $PROJECT"
echo "=============================================================="
