# Video Action Recognition - UCF101

A modular deep learning framework for video action recognition using the UCF101 dataset. This project implements multiple state-of-the-art architectures including CNNs, LSTM-based models, 3D convolutions, and ensemble methods.

## Project Structure

```
.
├── scripts/                    # Main source code
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Dataset loading utilities
│   ├── preprocessing.py       # Video frame extraction and preprocessing
│   ├── visualization.py       # Plotting and visualization functions
│   ├── training.py            # Training utilities and callbacks
│   ├── evaluation.py          # Model evaluation metrics
│   ├── utils.py               # General utility functions
│   ├── main.py                # Main entry point
│   └── models/                # Model architectures
│       ├── basic_cnn.py       # Basic CNN
│       ├── vgg_like.py        # VGG-like architecture
│       ├── mobilenet_transfer.py      # MobileNet transfer learning
│       ├── mobilenet_finetuned.py     # Fine-tuned MobileNet
│       ├── efficientnet.py    # EfficientNet with augmentation
│       ├── cnn_lstm.py        # CNN-LSTM temporal model
│       ├── pretrained_cnn_lstm.py     # Pretrained CNN-LSTM
│       ├── conv3d.py          # 3D Convolution model
│       ├── two_stream.py      # Two-stream architecture
│       └── ensemble.py        # Ensemble model
├── data/                      # Dataset storage
├── results/                   # Training results and metrics
│   └── checkpoints/          # Model checkpoints
├── visualizations/            # Generated plots and figures
├── logs/                      # Training logs
└── README.md
```

## Requirements

```bash
pip install tensorflow>=2.8.0 numpy pandas scikit-learn matplotlib seaborn plotly opencv-python pillow tqdm kagglehub
```

## Usage

### 1. Preprocess Dataset

Extract frames from UCF101 videos:

```bash
cd scripts
python main.py preprocess --dataset-path /path/to/ucf101 --percentage 0.2
```

Options:
- `--dataset-path`: Path to UCF101 dataset (downloads automatically if not provided)
- `--output-dir`: Output directory for extracted frames (default: /kaggle/working)
- `--percentage`: Percentage of videos to process per class (default: 0.2)

### 2. Train Models

Train a specific model:

```bash
python main.py train --model-type mobilenet --epochs 100 --batch-size 16 --evaluate
```

Available models:
- `basic_cnn`: Basic CNN architecture
- `vgg`: VGG-like deep CNN
- `mobilenet`: MobileNetV2 transfer learning
- `mobilenet_finetuned`: Fine-tuned MobileNetV2
- `efficientnet`: EfficientNetB0 with data augmentation
- `cnn_lstm`: CNN-LSTM for temporal modeling
- `pretrained_cnn_lstm`: Pretrained CNN-LSTM
- `conv3d`: 3D Convolutional network
- `two_stream`: Two-stream CNN
- `ensemble`: Ensemble of multiple models

Options:
- `--model-type`: Model architecture to train (required)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--evaluate`: Evaluate model on test set after training

### 3. Evaluate Models

Evaluate a trained model:

```bash
python main.py evaluate --model-name mobilenet
```

### 4. Visualize Data

Visualize dataset samples and distributions:

```bash
python main.py visualize
```

### 5. Distributed Training (Multi-GPU)

Train with DeepSpeed across multiple H100 GPUs:

```bash
# DDP on 8 GPUs
bash launch_distributed.sh mobilenet ddp 8 16 100

# FSDP Stage 2 on 8 GPUs
bash launch_distributed.sh efficientnet fsdp_stage2 8 16 100

# FSDP Stage 3 for large models
bash launch_distributed.sh ensemble fsdp_stage3 8 4 100
```

See [DEEPSPEED_GUIDE.md](DEEPSPEED_GUIDE.md) for detailed distributed training documentation.

## Configuration

Edit `scripts/config.py` to modify:
- Image size and preprocessing parameters
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and directories
- Model-specific settings

## Model Architectures

### Basic CNN
Simple convolutional architecture for baseline performance.

### VGG-Like
Deep CNN inspired by VGG architecture with multiple convolutional blocks.

### MobileNet Transfer Learning
Uses pretrained MobileNetV2 as feature extractor with custom classifier head.

### Fine-tuned MobileNet
MobileNetV2 with unfrozen top layers for domain-specific fine-tuning.

### EfficientNet with Augmentation
EfficientNetB0 with extensive data augmentation for improved generalization.

### CNN-LSTM
Combines CNN for spatial features with LSTM for temporal modeling.

### Pretrained CNN-LSTM
Uses pretrained MobileNetV2 features with LSTM for temporal sequences.

### 3D Convolution
3D CNN for joint spatio-temporal feature learning.

### Two-Stream
Separate spatial and temporal streams merged for final classification.

### Ensemble
Combines predictions from multiple architectures for improved accuracy.

## Results

Training results, checkpoints, and visualizations are saved to:
- `results/checkpoints/`: Model weights
- `results/*.json`: Evaluation metrics
- `visualizations/`: Training curves, confusion matrices, ROC curves
- `logs/`: Training logs

## Features

- Modular architecture for easy experimentation
- Multiple state-of-the-art models
- Automatic checkpoint saving and early stopping
- Comprehensive evaluation metrics
- Visualization utilities
- Progress tracking with tqdm
- GPU support with automatic memory growth
- **DeepSpeed integration for multi-GPU training** ⚡
  - DDP (Distributed Data Parallel)
  - FSDP (Fully Sharded Data Parallel) with ZeRO Stage 2 & 3
  - Multi-node training support
- SLURM cluster compatibility

## Dataset

This project uses the UCF101 dataset:
- 101 action classes
- 13,320 videos
- Average duration: ~7 seconds per video
- Resolution: 320x240

## License

MIT License

## Citation

If you use this code, please cite:
```
@misc{video_action_recognition_ucf101,
  title={Video Action Recognition - UCF101},
  author={Your Name},
  year={2025}
}
```

