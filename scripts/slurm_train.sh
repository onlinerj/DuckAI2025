#!/bin/bash
#SBATCH --job-name=ucf101_training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load python/3.9
module load cuda/11.8
module load cudnn

source venv/bin/activate

cd scripts

python main.py train --model-type $1 --epochs 100 --batch-size 16 --evaluate

echo "Training complete for $1"

