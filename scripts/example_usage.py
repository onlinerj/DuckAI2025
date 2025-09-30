#!/usr/bin/env python3
"""
Example usage demonstrating the API for the video action recognition framework.
"""

import config
import data_loader
import preprocessing
import visualization
import training
import evaluation
import utils
from models import create_mobilenet_transfer_model

def example_full_workflow():
    """Example: Complete workflow from data loading to evaluation"""
    
    print("=" * 60)
    print("Example: Full Training Workflow")
    print("=" * 60)
    
    utils.create_directories()
    
    print("\n1. Loading preprocessed data...")
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, train_filenames,
     val_images, val_labels, val_filenames,
     test_images, test_labels, test_filenames) = train_data
    
    print("\n2. Encoding labels...")
    (train_labels_encoded, val_labels_encoded, test_labels_encoded,
     train_labels_categorical, val_labels_categorical, test_labels_categorical,
     label_encoder, num_classes) = data_loader.encode_labels(
        train_labels, val_labels, test_labels
    )
    
    print("\n3. Creating model...")
    model = create_mobilenet_transfer_model(num_classes=num_classes)
    model = training.compile_model(model, num_classes=num_classes)
    utils.summarize_model(model)
    
    print("\n4. Training model...")
    history = training.train_model(
        model, train_images, train_labels_categorical,
        val_images, val_labels_categorical,
        model_name="example_mobilenet",
        epochs=5,
        batch_size=16
    )
    
    print("\n5. Evaluating model...")
    results = evaluation.full_evaluate_model(
        model, history, test_images, test_labels_categorical,
        test_labels_encoded, label_encoder.classes_,
        model_name="example_mobilenet"
    )
    
    print("\n6. Visualizing results...")
    visualization.visualize_images(
        test_images, test_labels, num_images=5,
        title="Test Samples",
        save_path=utils.get_visualization_path("example_test_samples.png")
    )
    
    print("\nWorkflow complete!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Visualizations saved to: {config.VISUALIZATIONS_DIR}")

def example_load_and_evaluate():
    """Example: Load a trained model and evaluate"""
    
    print("=" * 60)
    print("Example: Load and Evaluate Existing Model")
    print("=" * 60)
    
    import tensorflow as tf
    
    print("\n1. Loading data...")
    train_data = data_loader.load_processed_data()
    test_images, test_labels = train_data[6], train_data[7]
    
    print("\n2. Encoding labels...")
    _, _, test_labels_encoded, _, _, test_labels_categorical, label_encoder, _ = \
        data_loader.encode_labels(test_labels, test_labels, test_labels)
    
    print("\n3. Loading trained model...")
    model_path = utils.get_model_checkpoint_path("mobilenet")
    model = tf.keras.models.load_model(model_path)
    
    print("\n4. Evaluating...")
    results, predictions, predicted_classes = evaluation.evaluate_model(
        model, test_images, test_labels_categorical,
        test_labels_encoded, label_encoder.classes_,
        model_name="mobilenet"
    )
    
    print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")

def example_custom_training():
    """Example: Custom training loop with specific parameters"""
    
    from tensorflow.keras.callbacks import LearningRateScheduler
    import numpy as np
    
    print("=" * 60)
    print("Example: Custom Training Configuration")
    print("=" * 60)
    
    train_data = data_loader.load_processed_data()
    (train_images, train_labels, _,
     val_images, val_labels, _,
     test_images, test_labels, _) = train_data
    
    (train_labels_encoded, val_labels_encoded, test_labels_encoded,
     train_labels_categorical, val_labels_categorical, test_labels_categorical,
     label_encoder, num_classes) = data_loader.encode_labels(
        train_labels, val_labels, test_labels
    )
    
    from models import create_efficientnet_with_augmentation
    model = create_efficientnet_with_augmentation(num_classes=num_classes)
    
    model = training.compile_model(model, learning_rate=0.0001, num_classes=num_classes)
    
    def lr_schedule(epoch):
        lr = 0.0001
        if epoch > 50:
            lr *= 0.5
        if epoch > 75:
            lr *= 0.5
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = training.get_callbacks("custom_efficientnet", patience=15)
    callbacks.append(lr_scheduler)
    
    history = model.fit(
        train_images, train_labels_categorical,
        validation_data=(val_images, val_labels_categorical),
        epochs=10,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nCustom training complete!")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == 'full':
            example_full_workflow()
        elif example_name == 'evaluate':
            example_load_and_evaluate()
        elif example_name == 'custom':
            example_custom_training()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: full, evaluate, custom")
    else:
        print("Usage: python example_usage.py [full|evaluate|custom]")
        print("\nAvailable examples:")
        print("  full     - Complete workflow from data to evaluation")
        print("  evaluate - Load and evaluate existing model")
        print("  custom   - Custom training with specific parameters")

