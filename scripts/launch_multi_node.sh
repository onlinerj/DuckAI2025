#!/bin/bash

MODEL_TYPE=${1:-"mobilenet"}
STRATEGY=${2:-"fsdp_stage3"}
NUM_NODES=${3:-2}
NUM_GPUS_PER_NODE=${4:-8}
BATCH_SIZE=${5:-16}
EPOCHS=${6:-100}

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "============================================"
echo "Multi-Node Distributed Training"
echo "============================================"
echo "Model: $MODEL_TYPE"
echo "Strategy: $STRATEGY"
echo "Nodes: $NUM_NODES"
echo "GPUs per Node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * TOTAL_GPUS * 2))"
echo "Epochs: $EPOCHS"
echo "============================================"

deepspeed --num_nodes=$NUM_NODES \
    --num_gpus=$NUM_GPUS_PER_NODE \
    --hostfile hostfile.txt \
    train_distributed.py \
    --model-type $MODEL_TYPE \
    --strategy $STRATEGY \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps 2 \
    --learning-rate 0.001

echo "Multi-node training complete!"

