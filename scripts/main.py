import argparse
import sys
import os
import numpy as np
import tensorflow as tf

import config
import utils
import data_loader
import preprocessing
import visualization
import training
import evaluation
from models import *

def setup_environment():
    """Setup project environment"""
    utils.create_directories()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'GPU available: {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPU found, using CPU')

def preprocess_data(args):
    """Preprocess videos to extract frames"""
    print("Starting data preprocessing...")
    
    dataset_path = args.dataset_path
    if not dataset_path:
        dataset_path = data_loader.download_dataset()
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_path, split)
        print(f"\nProcessing {split} set...")
        
        preprocessing.process_videos_by_percentage(
            split_dir,
            percentage=args.percentage,
            sampling_rate=config.SAMPLING_RATE,
            output_base=args.output_dir
        )
        
        images, labels, filenames = preprocessing.load_images_with_filenames(
            f"/{split}",
            working_dir=args.output_dir,
            image_size=config.IMAGE_SIZE
        )
        
        preprocessing.save_processed_data(
            images, labels, filenames, split, config.DATA_DIR
        )
    
    print("\nPreprocessing complete!")

def train_single_model(args):
    """Train a single model"""
    print(f"Training model: {args.model_type}")
    
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, train_filenames,
     val_images, val_labels, val_filenames,
     test_images, test_labels, test_filenames) = train_data
    
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
    
    if args.model_type not in model_functions:
        print(f"Unknown model type: {args.model_type}")
        print(f"Available models: {list(model_functions.keys())}")
        return
    
    if args.model_type in ['cnn_lstm', 'pretrained_cnn_lstm', 'conv3d']:
        print("Creating sequences for temporal model...")
        train_sequences, train_seq_labels = preprocessing.create_sequences(
            train_images, train_labels_encoded, config.SEQUENCE_LENGTH
        )
        val_sequences, val_seq_labels = preprocessing.create_sequences(
            val_images, val_labels_encoded, config.SEQUENCE_LENGTH
        )
        
        from tensorflow.keras.utils import to_categorical
        train_seq_labels_cat = to_categorical(train_seq_labels, num_classes)
        val_seq_labels_cat = to_categorical(val_seq_labels, num_classes)
        
        model = model_functions[args.model_type](
            sequence_length=config.SEQUENCE_LENGTH,
            num_classes=num_classes
        )
        model = training.compile_model(model, num_classes=num_classes)
        utils.summarize_model(model)
        
        history = training.train_sequence_model(
            model, train_sequences, train_seq_labels_cat,
            val_sequences, val_seq_labels_cat,
            model_name=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        model = model_functions[args.model_type](num_classes=num_classes)
        model = training.compile_model(model, num_classes=num_classes)
        utils.summarize_model(model)
        
        history = training.train_model(
            model, train_images, train_labels_categorical,
            val_images, val_labels_categorical,
            model_name=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    print(f"\nModel {args.model_type} trained successfully!")
    
    if args.evaluate:
        print("\nEvaluating model...")
        results = evaluation.full_evaluate_model(
            model, history, test_images, test_labels_categorical,
            test_labels_encoded, label_encoder.classes_,
            model_name=args.model_type
        )

def evaluate_model(args):
    """Evaluate a trained model"""
    print(f"Evaluating model: {args.model_name}")
    
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, train_filenames,
     val_images, val_labels, val_filenames,
     test_images, test_labels, test_filenames) = train_data
    
    (train_labels_encoded, val_labels_encoded, test_labels_encoded,
     train_labels_categorical, val_labels_categorical, test_labels_categorical,
     label_encoder, num_classes) = data_loader.encode_labels(
        train_labels, val_labels, test_labels
    )
    
    model_path = utils.get_model_checkpoint_path(args.model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    results, predictions, predicted_classes = evaluation.evaluate_model(
        model, test_images, test_labels_categorical,
        test_labels_encoded, label_encoder.classes_,
        model_name=args.model_name
    )
    
    visualization.plot_confusion_matrix(
        test_labels_encoded, predicted_classes,
        label_encoder.classes_,
        title_prefix=args.model_name,
        save_path=utils.get_visualization_path(f"{args.model_name}_confusion_matrix.png")
    )
    
    utils.save_results(results, f"{args.model_name}_results.json")

def visualize_data(args):
    """Visualize dataset"""
    print("Visualizing dataset...")
    
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, train_filenames,
     val_images, val_labels, val_filenames,
     test_images, test_labels, test_filenames) = train_data
    
    visualization.print_class_distribution(train_labels, "Training")
    visualization.print_class_distribution(val_labels, "Validation")
    visualization.print_class_distribution(test_labels, "Test")
    
    visualization.visualize_images(
        train_images, train_labels, num_images=5,
        title="Training Samples",
        save_path=utils.get_visualization_path("train_samples.png")
    )
    
    visualization.visualize_images(
        val_images, val_labels, num_images=5,
        title="Validation Samples",
        save_path=utils.get_visualization_path("val_samples.png")
    )
    
    visualization.visualize_images(
        test_images, test_labels, num_images=5,
        title="Test Samples",
        save_path=utils.get_visualization_path("test_samples.png")
    )

def main():
    parser = argparse.ArgumentParser(description='Video Action Recognition - UCF101')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess videos')
    preprocess_parser.add_argument('--dataset-path', type=str, help='Path to UCF101 dataset')
    preprocess_parser.add_argument('--output-dir', type=str, default='/kaggle/working', 
                                   help='Output directory for frames')
    preprocess_parser.add_argument('--percentage', type=float, default=0.2,
                                   help='Percentage of videos to process per class')
    
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-type', type=str, required=True,
                             choices=['basic_cnn', 'vgg', 'mobilenet', 'mobilenet_finetuned',
                                    'efficientnet', 'cnn_lstm', 'pretrained_cnn_lstm',
                                    'conv3d', 'two_stream', 'ensemble'],
                             help='Model architecture to train')
    train_parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                             help='Batch size for training')
    train_parser.add_argument('--evaluate', action='store_true',
                             help='Evaluate model after training')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-name', type=str, required=True,
                            help='Name of the model to evaluate')
    
    viz_parser = subparsers.add_parser('visualize', help='Visualize dataset')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    setup_environment()
    
    if args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'train':
        train_single_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'visualize':
        visualize_data(args)

if __name__ == '__main__':
    main()

