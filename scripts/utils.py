import os
import numpy as np
import json
from datetime import datetime
import config

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        config.DATA_DIR,
        config.RESULTS_DIR,
        config.CHECKPOINTS_DIR,
        config.VISUALIZATIONS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories created")

def save_results(results_dict, filename):
    """Save results dictionary to JSON file"""
    filepath = os.path.join(config.RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, to_json_serializable(results_dict), indent=4)
    print(f"Results saved to {filepath}")

def to_json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def log_message(message, log_file="training.log"):
    """Log message to file with timestamp"""
    log_path = os.path.join(config.LOGS_DIR, log_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def get_model_checkpoint_path(model_name):
    """Get checkpoint path for a model"""
    return os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_best.keras")

def get_visualization_path(filename):
    """Get path for saving visualizations"""
    return os.path.join(config.VISUALIZATIONS_DIR, filename)

def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable = sum([np.prod(var.shape) for var in model.trainable_weights])
    non_trainable = sum([np.prod(var.shape) for var in model.non_trainable_weights])
    return trainable, non_trainable

def summarize_model(model):
    """Print concise model summary"""
    trainable, non_trainable = count_parameters(model)
    total = trainable + non_trainable
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Non-trainable: {non_trainable:,}")

