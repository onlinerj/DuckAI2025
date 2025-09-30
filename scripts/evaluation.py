import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.utils import plot_model
import visualization
import utils
import config

def top_k_accuracy(y_true, y_pred, k=5):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    correct = sum([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct / len(y_true)

def evaluate_model(model, test_images, test_labels_onehot, test_labels_encoded, 
                   class_names, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\nEvaluating {model_name}...")
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    top5_acc = top_k_accuracy(test_labels_encoded, predictions, k=5)
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels_encoded, predicted_classes, average='weighted', zero_division=0
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    results = {
        'model_name': model_name,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'top5_accuracy': float(top5_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return results, predictions, predicted_classes

def full_evaluate_model(model, history, test_images, test_labels_onehot, 
                       test_labels_encoded, class_names, model_name="Model"):
    """Full evaluation with visualizations"""
    print(f"\n{'='*50}")
    print(f"Full Evaluation: {model_name}")
    print(f"{'='*50}")
    
    utils.summarize_model(model)
    
    results, predictions, predicted_classes = evaluate_model(
        model, test_images, test_labels_onehot, test_labels_encoded, 
        class_names, model_name
    )
    
    save_prefix = model_name.lower().replace(' ', '_')
    
    visualization.plot_training_curves(
        history, 
        title_prefix=model_name,
        save_path=utils.get_visualization_path(f"{save_prefix}_training_curves.png")
    )
    
    visualization.plot_confusion_matrix(
        test_labels_encoded, 
        predicted_classes, 
        class_names,
        title_prefix=model_name,
        save_path=utils.get_visualization_path(f"{save_prefix}_confusion_matrix.png")
    )
    
    visualization.plot_roc_curves(
        test_labels_onehot, 
        predictions, 
        class_names,
        max_classes_to_plot=10,
        title_prefix=model_name,
        save_path=utils.get_visualization_path(f"{save_prefix}_roc_curves.png")
    )
    
    visualization.show_predictions(
        model, 
        test_images, 
        test_labels_encoded, 
        class_names,
        n=5,
        save_path=utils.get_visualization_path(f"{save_prefix}_predictions.png")
    )
    
    utils.save_results(results, f"{save_prefix}_results.json")
    
    print(f"\n{model_name} evaluation complete!")
    print(f"{'='*50}\n")
    
    return results

def generate_classification_report(y_true, y_pred, class_names, output_file=None):
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    if output_file:
        report_path = utils.get_visualization_path(output_file)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")
    
    return report

