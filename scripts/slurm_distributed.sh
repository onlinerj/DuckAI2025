#!/bin/bash
#SBATCH --job-name=ucf101_distributed
#SBATCH --output=../logs/slurm_distributed_%j.out
#SBATCH --error=../logs/slurm_distributed_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=gpu

MODEL_TYPE=${1:-"mobilenet"}
STRATEGY=${2:-"ddp"}
NUM_GPUS=${3:-8}
BATCH_SIZE=${4:-16}
EPOCHS=${5:-100}

module load python/3.9
module load cuda/12.1
module load cudnn/8.9
module load nccl/2.18

source ../venv/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$NUM_GPUS

echo "============================================"
echo "SLURM Distributed Training"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_TYPE"
echo "Strategy: $STRATEGY"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "============================================"

cd scripts

deepspeed --num_gpus=$NUM_GPUS train_distributed.py \
    --model-type $MODEL_TYPE \
    --strategy $STRATEGY \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps 2 \
    --learning-rate 0.001

echo "Training complete for $MODEL_TYPE with $STRATEGY strategy"

