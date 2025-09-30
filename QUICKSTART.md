# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create project directories:**
```bash
cd scripts
python -c "import utils; utils.create_directories()"
```

## Basic Workflow

### Option 1: Using Command Line Interface

```bash
cd scripts

# Step 1: Preprocess videos (extract frames)
python main.py preprocess --percentage 0.2

# Step 2: Visualize the dataset
python main.py visualize

# Step 3: Train a model
python main.py train --model-type mobilenet --epochs 50 --batch-size 16 --evaluate

# Step 4: Evaluate a trained model
python main.py evaluate --model-name mobilenet
```

### Option 2: Using Python API

```python
cd scripts
python example_usage.py full
```

## Training Specific Models

### Basic CNN (Fast, for testing)
```bash
python main.py train --model-type basic_cnn --epochs 50 --batch-size 32
```

### Transfer Learning (Recommended for quick results)
```bash
python main.py train --model-type mobilenet --epochs 100 --batch-size 16 --evaluate
```

### Deep CNN (Better accuracy, slower)
```bash
python main.py train --model-type efficientnet --epochs 100 --batch-size 8 --evaluate
```

### Temporal Models (Best for videos, requires sequences)
```bash
python main.py train --model-type cnn_lstm --epochs 100 --batch-size 8 --evaluate
```

## Running on SLURM Cluster

```bash
# Make script executable
chmod +x scripts/slurm_train.sh

# Submit job
sbatch scripts/slurm_train.sh mobilenet

# Check status
squeue -u $USER

# View output
tail -f logs/slurm_*.out
```

## Training Multiple Models

```bash
cd scripts
python train_all_models.py --epochs 50 --batch-size 16 \
    --models basic_cnn mobilenet efficientnet
```

## Project Structure After Setup

```
Video Action Recognition/
├── scripts/                   # Source code
├── data/                      # Preprocessed .npy files
├── results/
│   ├── checkpoints/          # Best model weights (.keras)
│   └── *.json                # Evaluation metrics
├── visualizations/           # Training curves, confusion matrices
└── logs/                     # Training logs
```

## Tips

1. **Start small:** Use `--percentage 0.1` and `--epochs 10` for quick testing
2. **Monitor GPU:** Check GPU usage with `nvidia-smi`
3. **Batch size:** Reduce if you encounter OOM errors
4. **Results:** All outputs are automatically saved to respective directories
5. **Resume training:** Models automatically save checkpoints

## Common Issues

### Out of Memory
```bash
# Reduce batch size
python main.py train --model-type mobilenet --batch-size 8
```

### Slow training
```bash
# Use lighter model or fewer epochs
python main.py train --model-type basic_cnn --epochs 30
```

### Dataset not found
```bash
# Specify dataset path explicitly
python main.py preprocess --dataset-path /path/to/ucf101
```

## Next Steps

1. Review results in `results/` directory
2. Check visualizations in `visualizations/` directory
3. Compare model performance using JSON files
4. Fine-tune best performing models
5. Try ensemble methods for improved accuracy

## Configuration

Edit `scripts/config.py` to modify:
- Default batch size and epochs
- Image resolution
- Data paths
- Model hyperparameters

## Distributed Training (Multi-GPU)

### Quick Start with DeepSpeed

Train on multiple H100 GPUs:

```bash
cd scripts

# DDP on 4 GPUs (fastest)
bash launch_distributed.sh mobilenet ddp 4 32 50

# FSDP Stage 2 on 8 GPUs (memory efficient)
bash launch_distributed.sh efficientnet fsdp_stage2 8 16 100

# FSDP Stage 3 on 8 GPUs (very large models)
bash launch_distributed.sh ensemble fsdp_stage3 8 4 100
```

### SLURM Cluster

```bash
# Submit distributed training job
sbatch slurm_distributed.sh mobilenet ddp 8 16 100

# Check job status
squeue -u $USER
```

For detailed information, see [DEEPSPEED_GUIDE.md](DEEPSPEED_GUIDE.md)

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages in terminal
3. Verify data is preprocessed correctly
4. Ensure GPU is available (if using)
5. For distributed training, see DeepSpeed guide

