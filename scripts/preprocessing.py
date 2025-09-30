import os
import cv2
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import config

def extract_frames(video_path, output_dir, sampling_rate=2):
    """Extract frames from a video at specified sampling rate"""
    try:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        os.makedirs(output_dir, exist_ok=True)
        
        num_frames_to_extract = math.ceil(duration / sampling_rate)
        frames_to_extract = [int(i * fps * sampling_rate) for i in range(num_frames_to_extract)]
        
        extracted_frames = []
        count = 0
        while count < frame_count:
            success, image = vidcap.read()
            if not success:
                break
            
            if count in frames_to_extract:
                timestamp = count / fps
                frame_name = os.path.join(output_dir, f"{video_filename}_frame_{int(timestamp)}sec.jpg")
                cv2.imwrite(frame_name, image)
                extracted_frames.append(frame_name)
            
            count += 1
        
        vidcap.release()
        return extracted_frames
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []

def process_videos_by_percentage(base_dir, percentage=0.2, sampling_rate=2, output_base="/"):
    """Process a percentage of videos from each action class"""
    action_classes = [item for item in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, item))]
    
    print(f"Found {len(action_classes)} action classes in {base_dir}")
    
    for action_class in tqdm(action_classes, desc="Processing action classes"):
        action_class_path = os.path.join(base_dir, action_class)
        
        video_files = [os.path.join(action_class_path, f) 
                      for f in os.listdir(action_class_path)
                      if f.endswith(('.avi', '.mp4'))]
        video_files.sort()
        
        num_to_process = max(1, int(len(video_files) * percentage))
        videos_to_process = video_files[:num_to_process]
        
        for video_path in videos_to_process:
            output_dir = os.path.join(output_base, base_dir.lstrip("/"), action_class, "frames")
            os.makedirs(output_dir, exist_ok=True)
            extract_frames(video_path, output_dir, sampling_rate)

def load_images_with_filenames(base_folder, working_dir="/", image_size=(224, 224)):
    """Load images, labels, and filenames from extracted frames"""
    images = []
    labels = []
    filenames = []
    
    print(f"Loading frames from: {base_folder}")
    
    action_dirs = [item for item in os.listdir(base_folder) 
                   if os.path.isdir(os.path.join(base_folder, item))]
    
    print(f"Found {len(action_dirs)} action class directories")
    
    for action in tqdm(action_dirs, desc="Loading frames"):
        frames_dir = os.path.join(working_dir, base_folder.lstrip("/"), action, "frames")
        
        if not os.path.exists(frames_dir):
            continue
        
        frame_files = [os.path.join(frames_dir, f) 
                      for f in os.listdir(frames_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for frame_path in frame_files:
            try:
                img = Image.open(frame_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(action)
                filenames.append(os.path.basename(frame_path))
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
    
    images_array = np.array(images) if images else np.empty((0, *image_size, 3))
    labels_array = np.array(labels)
    filenames_array = np.array(filenames)
    
    print(f"Loaded {len(images)} frames with {len(set(labels))} unique action classes")
    
    return images_array, labels_array, filenames_array

def create_sequences(images, labels, sequence_length=5):
    """Create sequences for temporal models (LSTM, Conv3D)"""
    sequences = []
    sequence_labels = []
    
    unique_labels = np.unique(labels)
    
    for label in tqdm(unique_labels, desc="Creating sequences"):
        label_indices = np.where(labels == label)[0]
        
        for i in range(0, len(label_indices) - sequence_length + 1, sequence_length):
            sequence_indices = label_indices[i:i + sequence_length]
            
            if len(sequence_indices) == sequence_length:
                sequence = images[sequence_indices]
                sequences.append(sequence)
                sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)

def save_processed_data(images, labels, filenames, dataset_type, target_dir):
    """Save processed data to .npy files"""
    os.makedirs(target_dir, exist_ok=True)
    
    np.save(os.path.join(target_dir, f"{dataset_type}_images.npy"), images)
    np.save(os.path.join(target_dir, f"{dataset_type}_labels.npy"), labels)
    np.save(os.path.join(target_dir, f"{dataset_type}_filenames.npy"), filenames)
    
    print(f"{dataset_type} data saved:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Filenames: {filenames.shape}")

