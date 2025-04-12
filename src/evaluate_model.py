import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

def evaluate_model(predictions, true_labels):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    predictions : array-like
        Predicted labels
    true_labels : array-like
        True labels
        
    Returns:
    --------
    accuracy : float
        Model accuracy
    """
    accuracy = np.mean(predictions == true_labels)
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy

def print_confusion_matrix(true_labels, predictions):
    """
    Print confusion matrix
    
    Parameters:
    -----------
    true_labels : array-like
        True labels
    predictions : array-like
        Predicted labels
    """
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion matrix:")
    print(cm)
    return cm

def print_classification_report(true_labels, predictions, target_names=None):
    """
    Print classification report
    
    Parameters:
    -----------
    true_labels : array-like
        True labels
    predictions : array-like
        Predicted labels
    target_names : list, default=None
        List of target class names
    """
    if target_names is None:
        target_names = np.unique(true_labels)
    
    report = classification_report(true_labels, predictions, target_names=target_names)
    print("Classification report:")
    print(report)
    return report

def plot_confusion_matrix(true_labels, predictions, target_names=None, figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    true_labels : array-like
        True labels
    predictions : array-like
        Predicted labels
    target_names : list, default=None
        List of target class names
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : str, default=None
        Path to save the figure
    """
    if target_names is None:
        target_names = np.unique(true_labels)
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, figsize=(10, 5), save_path=None):
    """
    Plot class distribution
    
    Parameters:
    -----------
    labels : array-like
        Class labels
    figsize : tuple, default=(10, 5)
        Figure size
    save_path : str, default=None
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Count occurrences of each class
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Sort by counts
    sorted_indices = np.argsort(counts)[::-1]
    unique_classes = unique_classes[sorted_indices]
    counts = counts[sorted_indices]
    
    # Plot
    bars = plt.bar(unique_classes, counts)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()

def compare_models(model_names, accuracies, figsize=(8, 5), save_path=None):
    """
    Compare multiple models based on accuracy
    
    Parameters:
    -----------
    model_names : list
        List of model names
    accuracies : list
        List of model accuracies
    figsize : tuple, default=(8, 5)
        Figure size
    save_path : str, default=None
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot
    bars = plt.bar(model_names, accuracies, color='skyblue')
    
    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show() 