#!/bin/bash

MODEL_TYPE=${1:-"mobilenet"}
STRATEGY=${2:-"ddp"}
NUM_GPUS=${3:-4}
BATCH_SIZE=${4:-16}
EPOCHS=${5:-100}

echo "============================================"
echo "Launching Distributed Training"
echo "============================================"
echo "Model: $MODEL_TYPE"
echo "Strategy: $STRATEGY"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "============================================"

deepspeed --num_gpus=$NUM_GPUS train_distributed.py \
    --model-type $MODEL_TYPE \
    --strategy $STRATEGY \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps 1 \
    --learning-rate 0.001

echo "Training complete!"

