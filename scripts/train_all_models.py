#!/usr/bin/env python3
"""
Script to train all models sequentially.
Useful for batch training on SLURM clusters.
"""

import os
import sys
from main import setup_environment, train_single_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train all models sequentially')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--models', nargs='+', 
                       default=['basic_cnn', 'vgg', 'mobilenet', 'efficientnet'],
                       help='List of models to train')
    
    args = parser.parse_args()
    
    setup_environment()
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Training: {model_type}")
        print(f"{'='*60}\n")
        
        model_args = argparse.Namespace(
            model_type=model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            evaluate=True
        )
        
        try:
            train_single_model(model_args)
            print(f"\n✓ {model_type} completed successfully")
        except Exception as e:
            print(f"\n✗ {model_type} failed with error: {e}")
            continue
    
    print("\n" + "="*60)
    print("All models training complete!")
    print("="*60)

if __name__ == '__main__':
    main()

