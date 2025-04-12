import numpy as np
import yaml
from typing import List, Dict, Any, Sequence, Tuple
import tensorflow as tf
from tensorflow import keras
from transformers import DistilBertTokenizer
import os
import pickle
import json
import pandas as pd
from sklearn.pipeline import Pipeline
import re

def load_dataset_dl(file_path, text_column, label_column, max_length, tokenizer=None, unique_labels=None):
    """Load and process a dataset file with consolidated categories"""
    # Try different delimiters
    try:
        df = pd.read_csv(file_path, delimiter=";")
    except:
        try:
            df = pd.read_csv(file_path, delimiter=",")
        except Exception as e:
            raise ValueError(f"Could not read file {file_path} with delimiter ',' or ';': {e}")
    
    print(f"Loaded {file_path} with {len(df)} rows")
    
    # Get text and labels
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()
    
    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        tokenizer.max_length = max_length
        tokenizer.pad_to_max_length = True

    # Get unique labels if not provided\n",
    if unique_labels is None:
      unique_labels = sorted(list(set(labels)))
    
    # Encode labels
    encoded_labels = encode_labels(labels, unique_labels)
    
    # Encode texts
    print(f"\nTokenizing texts from {file_path}...")
    encoded_texts = encode_texts(tokenizer, texts)
    
    return encoded_texts, encoded_labels, tokenizer, unique_labels


def load_dataset_ml(file_path, text_column, label_column, unique_labels=None):
    """
    Load and process a dataset file for traditional ML models without preprocessing
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    text_column : str
        Name of the column containing text data
    label_column : str
        Name of the column containing labels
    unique_labels : list, default=None
        List of unique labels. If None, it will be inferred from the data
        
    Returns:
    --------
    texts : list
        List of raw texts
    labels : list
        List of labels
    unique_labels : list
        List of unique labels
    """
    # Try different delimiters
    try:
        df = pd.read_csv(file_path, delimiter=";")
    except:
        try:
            df = pd.read_csv(file_path, delimiter=",")
        except Exception as e:
            raise ValueError(f"Could not read file {file_path} with delimiter ',' or ';': {e}")
    
    print(f"Loaded {file_path} with {len(df)} rows")
    
    # Get text and labels
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()
    
    # Get unique labels if not provided
    if unique_labels is None:
        unique_labels = sorted(list(set(labels)))
    
    return texts, labels, unique_labels


def load_datasets_for_ml(train_path, valid_path=None, test_path=None, 
                       text_column='text_en', label_column='queue'):
    """
    Load datasets specifically for traditional ML models without preprocessing
    
    Parameters:
    -----------
    train_path : str
        Path to the training dataset
    valid_path : str, default=None
        Path to the validation dataset
    test_path : str, default=None
        Path to the test dataset
    text_column : str, default='text_en'
        Name of the column containing text data
    label_column : str, default='queue'
        Name of the column containing labels
        
    Returns:
    --------
    datasets : dict
        Dictionary containing train, validation, and test data
    """
    print("Loading datasets for traditional ML models...")
    datasets = {}
    
    # Load training data
    train_texts, train_labels, unique_labels = load_data_ml(
        train_path, text_column, label_column
    )
    datasets['train'] = {
        'texts': train_texts,
        'labels': train_labels
    }
    
    # Load validation data if provided
    if valid_path:
        valid_texts, valid_labels, _ = load_data_ml(
            valid_path, text_column, label_column, 
            unique_labels=unique_labels
        )
        datasets['valid'] = {
            'texts': valid_texts,
            'labels': valid_labels
        }
    
    # Load test data if provided
    if test_path:
        test_texts, test_labels, _ = load_data_ml(
            test_path, text_column, label_column, 
            unique_labels=unique_labels
        )
        datasets['test'] = {
            'texts': test_texts,
            'labels': test_labels
        }
    
    # Store unique labels
    datasets['unique_labels'] = unique_labels
    
    return datasets


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
        return get_config()


def get_config():
    """
    Returns the default configuration for model training
    
    Returns:
    --------
    config : dict
        Default configuration dictionary
    """
    return {
        "data": {
            "train_path": "data/ticket_train.csv",
            "valid_path": "data/ticket_valid.csv",
            "test_path": "data/ticket_test.csv",
            "text_column": "text",
            "label_column": "queue",
            "max_words_per_message": 128,
            "pad_to_max_length": True
        },
        "training": {
            "learning_rate": 5e-5,
            "batch_size": 32,
            "epochs": 3,
            "early_stopping_patience": 2,
            "early_stopping_min_delta_acc": 0.01
        }
    }


def update_config(config, **kwargs):
    """
    Update the configuration with new values
    
    Parameters:
    -----------
    config : dict
        Original configuration dictionary
    **kwargs : dict
        Key-value pairs to update in the configuration
        
    Returns:
    --------
    config : dict
        Updated configuration
    """
    for key, value in kwargs.items():
        if key in config and isinstance(value, dict) and isinstance(config[key], dict):
            # If both are dictionaries, recursively update
            config[key].update(value)
        else:
            # Otherwise, replace the value
            config[key] = value
    
    return config

def save_model(model, model_path):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : Pipeline or GridSearchCV
        Trained model to save
    model_path : str
        Path to save the model
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    model : Pipeline or GridSearchCV
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model

def save_config(config, config_path):
    """
    Save configuration to disk
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save the configuration
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")

def load_config(config_path):
    """
    Load configuration from disk
    
    Parameters:
    -----------
    config_path : str
        Path to the saved configuration
        
    Returns:
    --------
    config : dict
        Loaded configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config

def create_experiment_dir(base_dir='experiments', experiment_name=None):
    """
    Create a directory for an experiment
    
    Parameters:
    -----------
    base_dir : str, default='experiments'
        Base directory for experiments
    experiment_name : str, default=None
        Name of the experiment. If None, a timestamp will be used.
        
    Returns:
    --------
    experiment_dir : str
        Path to the experiment directory
    """
    import datetime
    
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    return experiment_dir

def get_sample_texts(df, text_column, n_samples=5, seed=42):
    """
    Get sample texts from a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Name of the column containing texts
    n_samples : int, default=5
        Number of samples to return
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    samples : list
        List of sample texts
    """
    np.random.seed(seed)
    sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
    samples = df.iloc[sample_indices][text_column].tolist()
    return samples

def predict_text(model, text, vectorizer=None):
    """
    Predict the class of a text
    
    Parameters:
    -----------
    model : Pipeline or GridSearchCV
        Trained model
    text : str
        Input text
    vectorizer : CountVectorizer, default=None
        Vectorizer instance. Only needed if model is not a Pipeline.
        
    Returns:
    --------
    prediction : str
        Predicted class
    """
    if isinstance(model, Pipeline):
        # If model is a pipeline, it already contains the vectorizer
        prediction = model.predict([text])[0]
    else:
        # If model is not a pipeline, vectorize the text first
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided if model is not a Pipeline")
        
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]
    
    return prediction

def print_runtime_info():
    """
    Print runtime information about the Python environment
    """
    import sys
    import platform
    import sklearn
    import numpy
    import pandas
    import scipy
    
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("Scikit-learn version:", sklearn.__version__)
    print("NumPy version:", numpy.__version__)
    print("Pandas version:", pandas.__version__)
    print("SciPy version:", scipy.__version__)
    
    try:
        import nltk
        print("NLTK version:", nltk.__version__)
    except ImportError:
        print("NLTK not installed")
    
    try:
        import matplotlib
        print("Matplotlib version:", matplotlib.__version__)
    except ImportError:
        print("Matplotlib not installed")
    
    try:
        import seaborn
        print("Seaborn version:", seaborn.__version__)
    except ImportError:
        print("Seaborn not installed") 