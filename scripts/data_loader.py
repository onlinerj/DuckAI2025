import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import config

def download_dataset():
    """Download UCF101 dataset from Kaggle"""
    import kagglehub
    path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")
    print(f"Dataset downloaded to: {path}")
    return path

def load_csv_splits(data_dir):
    """Load train, test, and validation CSV files"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    print(f"Train data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"Validation data: {val_df.shape}")
    
    return train_df, test_df, val_df

def load_processed_data(data_dir=None):
    """Load preprocessed .npy files"""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    print("Loading datasets...")
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_images = np.load(os.path.join(data_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(data_dir, "val_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    
    try:
        train_filenames = np.load(os.path.join(data_dir, "train_filenames.npy"))
        val_filenames = np.load(os.path.join(data_dir, "val_filenames.npy"))
        test_filenames = np.load(os.path.join(data_dir, "test_filenames.npy"))
    except:
        train_filenames = None
        val_filenames = None
        test_filenames = None
    
    print(f"Train set: {train_images.shape}")
    print(f"Validation set: {val_images.shape}")
    print(f"Test set: {test_images.shape}")
    
    return (train_images, train_labels, train_filenames,
            val_images, val_labels, val_filenames,
            test_images, test_labels, test_filenames)

def encode_labels(train_labels, val_labels, test_labels):
    """Encode string labels to integers and one-hot"""
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    num_classes = len(label_encoder.classes_)
    train_labels_categorical = to_categorical(train_labels_encoded, num_classes)
    val_labels_categorical = to_categorical(val_labels_encoded, num_classes)
    test_labels_categorical = to_categorical(test_labels_encoded, num_classes)
    
    print(f"Number of classes: {num_classes}")
    
    return (train_labels_encoded, val_labels_encoded, test_labels_encoded,
            train_labels_categorical, val_labels_categorical, test_labels_categorical,
            label_encoder, num_classes)

def analyze_video_durations(base_path):
    """Analyze video durations in the dataset"""
    import cv2
    durations = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                video_path = os.path.join(root, file)
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    durations.append(duration)
                    cap.release()
                except Exception as e:
                    print(f"Error analyzing {video_path}: {e}")
    return durations

def analyze_videos(root_dir, num_samples=5):
    """Analyze video metadata (resolution, frame rate, duration)"""
    import cv2
    import random
    
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.avi', '.mp4')):
                video_files.append(os.path.join(dirpath, filename))
    
    sampled_videos = random.sample(video_files, min(num_samples, len(video_files)))
    
    summary_data = []
    for video_file in sampled_videos:
        try:
            vidcap = cv2.VideoCapture(video_file)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            summary_data.append({
                'File': video_file,
                'Resolution': f"{width}x{height}",
                'Frame Rate': fps,
                'Duration': duration,
                'Frame Count': frame_count
            })
            vidcap.release()
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
    
    return pd.DataFrame(summary_data)

