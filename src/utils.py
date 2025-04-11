import numpy as np
import yaml
from typing import List, Dict, Any, Sequence
import tensorflow as tf
from tensorflow import keras
from transformers import DistilBertTokenizer


def encode_texts(tokenizer: DistilBertTokenizer, texts: Sequence[str]) -> Dict[str, tf.Tensor]:
    """
    Encodes a list of texts using the DistilBERT tokenizer.
    
    Args:
        tokenizer: The DistilBERT tokenizer to use
        texts: List of text strings to encode
        
    Returns:
        Dictionary with input_ids, attention_mask and other tokenizer outputs
    """
    encoded_texts = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=tokenizer.max_length if hasattr(tokenizer, "max_length") else 128,
        padding="max_length" if hasattr(tokenizer, "pad_to_max_length") and tokenizer.pad_to_max_length else "longest",
        truncation=True,
        return_tensors="tf"
    )
    return encoded_texts


def encode_labels(labels: List[str], unique_labels: List[str]) -> np.ndarray:
    """
    One-hot encodes labels.
    
    Args:
        labels: List of label strings
        unique_labels: List of all possible unique labels in the dataset
        
    Returns:
        One-hot encoded labels as a numpy array
    """
    label_indices = [unique_labels.index(label) for label in labels]
    one_hot_labels = keras.utils.to_categorical(label_indices, num_classes=len(unique_labels))
    return one_hot_labels


def load_training_conf(conf_path: str = "config/training_conf.yaml") -> Dict[str, Any]:
    """
    Loads the training configuration from a YAML file.
    
    Args:
        conf_path: Path to the config file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(conf_path, "r") as conf_file:
            conf = yaml.safe_load(conf_file)
        return conf
    except Exception as e:
        print(f"Error loading config file: {e}")
        # Return default config
        return {
            "data": {
                "dataset_path": "data/tickets.csv",
                "text_column": "text",
                "label_column": "queue",
                "max_words_per_message": 128,
                "pad_to_max_length": True
            },
            "training": {
                "test_set_size": 0.2,
                "learning_rate": 5e-5,
                "batch_size": 32,
                "epochs": 3,
                "early_stopping_patience": 2,
                "early_stopping_min_delta_acc": 0.01
            }
        } 