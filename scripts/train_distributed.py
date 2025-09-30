#!/usr/bin/env python3
"""
Distributed training script using DeepSpeed for multi-GPU training.
Supports both DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel).
"""

import argparse
import os
import sys
import torch

import config
import utils
import data_loader
import distributed_training
from models import *

def main():
    parser = argparse.ArgumentParser(description='Distributed Training with DeepSpeed')
    
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['basic_cnn', 'vgg', 'mobilenet', 'mobilenet_finetuned',
                              'efficientnet', 'cnn_lstm', 'pretrained_cnn_lstm',
                              'conv3d', 'two_stream', 'ensemble'],
                       help='Model architecture to train')
    
    parser.add_argument('--strategy', type=str, default='ddp',
                       choices=['ddp', 'fsdp_stage2', 'fsdp_stage3'],
                       help='Distributed training strategy')
    
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size per GPU')
    
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by launcher)')
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = distributed_training.setup_distributed()
    
    if distributed_training.is_main_process(rank):
        utils.create_directories()
        print(f"\n{'='*60}")
        print(f"Distributed Training Configuration")
        print(f"{'='*60}")
        print(f"Model: {args.model_type}")
        print(f"Strategy: {args.strategy}")
        print(f"World Size: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"{'='*60}\n")
    
    if distributed_training.is_main_process(rank):
        print("Loading datasets...")
    
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, _,
     val_images, val_labels, _,
     test_images, test_labels, _) = train_data
    
    (train_labels_encoded, val_labels_encoded, test_labels_encoded,
     train_labels_categorical, val_labels_categorical, test_labels_categorical,
     label_encoder, num_classes) = data_loader.encode_labels(
        train_labels, val_labels, test_labels
    )
    
    model_functions = {
        'basic_cnn': create_basic_cnn,
        'vgg': create_vgg_like_model,
        'mobilenet': create_mobilenet_transfer_model,
        'mobilenet_finetuned': create_mobilenet_finetuned_model,
        'efficientnet': create_efficientnet_with_augmentation,
        'cnn_lstm': create_cnn_lstm_model,
        'pretrained_cnn_lstm': create_pretrained_cnn_lstm_model,
        'conv3d': create_conv3d_model,
        'two_stream': create_two_stream_model,
        'ensemble': create_ensemble_model
    }
    
    if distributed_training.is_main_process(rank):
        print(f"Creating model: {args.model_type}...")
    
    model = model_functions[args.model_type](num_classes=num_classes)
    
    if distributed_training.is_main_process(rank):
        utils.summarize_model(model)
    
    train_dataset = distributed_training.create_torch_dataset(
        train_images, train_labels_categorical
    )
    val_dataset = distributed_training.create_torch_dataset(
        val_images, val_labels_categorical
    )
    
    if distributed_training.is_main_process(rank):
        print(f"\nStarting distributed training with {args.strategy}...")
    
    history = distributed_training.train_with_deepspeed(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=f"{args.model_type}_{args.strategy}",
        strategy=args.strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate
    )
    
    if distributed_training.is_main_process(rank):
        print("\nTraining complete!")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        
        results = {
            'model_type': args.model_type,
            'strategy': args.strategy,
            'world_size': world_size,
            'batch_size_per_gpu': args.batch_size,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps * world_size,
            'best_val_accuracy': float(max(history['val_accuracy'])),
            'final_train_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1])
        }
        
        utils.save_results(results, f"{args.model_type}_{args.strategy}_results.json")
    
    distributed_training.cleanup_distributed()

if __name__ == '__main__':
    main()

