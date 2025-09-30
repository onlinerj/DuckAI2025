import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

def plot_label_distribution(train_df, save_path=None):
    """Visualize dataset label distribution using treemap"""
    label_counts = train_df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    total_labels = len(label_counts)
    
    fig = px.treemap(label_counts, path=['label'], values='count',
                     color='count', hover_data=['count'],
                     color_continuous_scale='RdBu',
                     title=f"Dataset Labels Distribution (Total Labels: {total_labels})")
    fig.update_traces(textinfo="label+value")
    
    if save_path:
        fig.write_html(save_path)
    fig.show()

def visualize_video_summary(video_summary, save_path=None):
    """Visualize video metadata statistics"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Frame Rate Distribution", "Duration Distribution", 
                                       "Frame Count Distribution", "Resolution Distribution"))
    
    fig.add_trace(go.Histogram(x=video_summary['Frame Rate'], nbinsx=20), row=1, col=1)
    fig.add_trace(go.Histogram(x=video_summary['Duration'], nbinsx=20), row=1, col=2)
    fig.add_trace(go.Histogram(x=video_summary['Frame Count'], nbinsx=20), row=2, col=1)
    
    resolution_counts = video_summary['Resolution'].value_counts().reset_index()
    resolution_counts.columns = ['Resolution', 'Count']
    fig.add_trace(go.Bar(x=resolution_counts['Resolution'], y=resolution_counts['Count']), row=2, col=2)
    
    fig.update_layout(height=800, width=1000, title_text="Video Summary Visualization")
    
    if save_path:
        fig.write_html(save_path)
    fig.show()

def visualize_random_frames(root_dir, num_images=5, dataset_type=None, save_path=None):
    """Visualize random frames from extracted frames"""
    import cv2
    
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "frames":
                frames_dir = os.path.join(dirpath, dirname)
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(frames_dir):
                    for filename in sub_filenames:
                        if filename.endswith('.jpg'):
                            image_files.append(os.path.join(sub_dirpath, filename))
    
    if not image_files:
        print("No image files found")
        return
    
    sampled_images = random.sample(image_files, min(num_images, len(image_files)))
    
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(sampled_images):
        img = mpimg.imread(image_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        
        title = image_path.split(os.sep)[-3]
        if dataset_type:
            title = f"{dataset_type}: {title}"
        
        plt.title(title)
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def visualize_images(images, labels, num_images=5, title=None, save_path=None):
    """Visualize sample images from loaded dataset"""
    if len(images) == 0:
        print(f"No images to visualize for {title}")
        return
    
    plt.figure(figsize=(15, 3))
    random_indices = random.sample(range(len(images)), min(num_images, len(images)))
    
    for i, index in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[index])
        plt.title(labels[index])
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def print_class_distribution(labels, dataset_name):
    """Print class distribution statistics"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n{dataset_name} Class Distribution:")
    print(f"Total classes: {len(unique_labels)}")
    print(f"Top 10 classes by frequency:")
    
    sorted_indices = np.argsort(-counts)
    for i in range(min(10, len(unique_labels))):
        idx = sorted_indices[i]
        print(f"  {unique_labels[idx]}: {counts[idx]} frames")

def plot_training_curves(history, title_prefix="", save_path=None):
    """Plot training and validation metrics"""
    epochs = range(1, len(history.history['accuracy']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_acc = 'tab:blue'
    color_loss = 'tab:red'
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color_acc)
    ax1.plot(epochs, history.history['accuracy'], label='Training Accuracy', 
             color=color_acc, linestyle='-')
    ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', 
             color=color_acc, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 1.05)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color=color_loss)
    ax2.plot(epochs, history.history['loss'], label='Training Loss', 
             color=color_loss, linestyle='-')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss', 
             color=color_loss, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.set_ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1)
    
    fig.suptitle(f'{title_prefix} Training and Validation Metrics')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title_prefix="", save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title_prefix} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_roc_curves(y_true_onehot, y_scores, class_names, max_classes_to_plot=10, 
                   title_prefix="", save_path=None):
    """Plot ROC curves for top classes"""
    plt.figure(figsize=(10, 8))
    
    class_indices = random.sample(range(len(class_names)), 
                                 min(max_classes_to_plot, len(class_names)))
    
    for i in class_indices:
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix} ROC Curves (Sample of {max_classes_to_plot} classes)')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def show_predictions(model, images, labels_true, class_names, n=5, save_path=None):
    """Show model predictions on sample images"""
    sample_indices = random.sample(range(len(images)), min(n, len(images)))
    sample_images = images[sample_indices]
    sample_labels = [labels_true[i] for i in sample_indices]
    
    predictions = model.predict(sample_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(len(sample_images)):
        plt.subplot(1, n, i + 1)
        plt.imshow(sample_images[i])
        true_label = class_names[sample_labels[i]]
        pred_label = class_names[predicted_classes[i]]
        color = 'green' if sample_labels[i] == predicted_classes[i] else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

