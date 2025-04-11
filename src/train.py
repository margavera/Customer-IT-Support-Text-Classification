import os
from typing import Optional, List, Tuple, Dict

import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import tensorflow as tf
from tensorflow import keras
import gc

from src.utils import encode_labels, encode_texts, load_training_conf
from src.model import DistilBertClassifier, save_model

# Use GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def training_data(
    tickets_data_path: str,
    text_column: str,
    label_column: str,
    test_size: float = 0.25,
    subset_size: int = -1,
    max_length: int = 100,
    pad_to_max_length: bool = True,
) -> Tuple[Tuple[dict, dict, np.ndarray, np.ndarray], DistilBertTokenizer, List[str]]:
    """
    Prepares training and test data for the model.
    
    Args:
        tickets_data_path: Path to the CSV file containing the data
        text_column: Name of the column containing the text data
        label_column: Name of the column containing the labels
        test_size: Proportion of the data to use for testing
        subset_size: Number of samples to use (-1 for all)
        max_length: Maximum length of tokenized sequences
        pad_to_max_length: Whether to pad sequences to max_length
        
    Returns:
        Tuple containing (x_train, x_test, y_train, y_test), tokenizer, and unique_labels
    """
    # Try with multiple delimiters
    try:
        df = pd.read_csv(tickets_data_path, delimiter=";")
    except:
        try:
            df = pd.read_csv(tickets_data_path, delimiter=",")
        except:
            raise ValueError(f"Could not read file {tickets_data_path} with delimiter ',' or ';'")
    
    print(f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
    
    x = df[text_column].tolist()
    y = df[label_column].tolist()
    unique_labels = sorted(list(set(y)))
    
    # Encode the labels
    y = encode_labels(y, unique_labels)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tokenizer.max_length = max_length
    tokenizer.pad_to_max_length = pad_to_max_length
    
    print("Tokenizing all texts...")
    x = encode_texts(tokenizer, x)
    
    # Convert tensors to numpy arrays for splitting
    input_ids_np = x["input_ids"].numpy()
    attention_mask_np = x["attention_mask"].numpy()
    
    # Use full dataset or subset
    subset_size = len(input_ids_np) if subset_size < 0 else subset_size
    
    # Split data into train and test sets
    indices = np.arange(len(input_ids_np))
    train_indices, test_indices = train_test_split(
        indices[:subset_size], test_size=test_size, random_state=42
    )
    
    # Extract train and test data
    x_train = {
        "input_ids": tf.convert_to_tensor(input_ids_np[train_indices]),
        "attention_mask": tf.convert_to_tensor(attention_mask_np[train_indices])
    }
    x_test = {
        "input_ids": tf.convert_to_tensor(input_ids_np[test_indices]),
        "attention_mask": tf.convert_to_tensor(attention_mask_np[test_indices])
    }
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Clear memory
    del input_ids_np
    del attention_mask_np
    gc.collect()
    
    return (x_train, x_test, y_train, y_test), tokenizer, unique_labels


def define_callbacks(
    patience: int = 3, min_delta: float = 0.01
) -> List:
    """
    Defines callbacks for model training.
    
    Args:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        List of callbacks
    """
    early_stopper = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=min_delta,
        patience=patience,
        verbose=1,
        restore_best_weights=True,
    )
    
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(
            "logs", "scalars", datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        )
    )
    
    # Add memory cleanup callback
    class MemoryCleanupCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()
    
    return [early_stopper, tensorboard, MemoryCleanupCallback()]


def train_model(
    model: DistilBertClassifier,
    x_train: dict,
    x_test: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 16,
    epochs: int = 3,
    callbacks: Optional[List] = None,
    eval_batch_size: int = 16,
    class_weight: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    Trains the model.
    
    Args:
        model: The DistilBERT classifier model
        x_train: Training inputs
        x_test: Test inputs
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        callbacks: List of callbacks to use during training
        eval_batch_size: Batch size for evaluation
        class_weight: Dictionary of class weights
        
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    # Create dataset and batches for training
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": x_train["input_ids"], "attention_mask": x_train["attention_mask"]}, 
        y_train
    )).shuffle(buffer_size=1000).batch(batch_size)
    
    # Dataset for validation during training
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": x_test["input_ids"], "attention_mask": x_test["attention_mask"]}, 
        y_test
    )).batch(eval_batch_size)
    
    print(f"Training with batch size: {batch_size}, evaluation batch size: {eval_batch_size}")
    print(f"Training on {len(y_train)} samples, validating on {len(y_test)} samples")
    
    # Train the model with class weights
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    
    # Free up memory before evaluation
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Evaluate the model in batches
    print("Evaluating model in batches...")
    total_loss = 0
    total_accuracy = 0
    total_batches = 0
    
    for batch in val_dataset:
        x_batch, y_batch = batch
        batch_loss, batch_accuracy = model.evaluate(x_batch, y_batch, verbose=0)
        total_loss += batch_loss
        total_accuracy += batch_accuracy
        total_batches += 1
    
    # Calculate average metrics
    test_loss = total_loss / total_batches
    test_accuracy = total_accuracy / total_batches
    
    return test_loss, test_accuracy


if __name__ == "__main__":
    # Load configuration
    conf = load_training_conf()
    conf_train, conf_data = conf["training"], conf["data"]
    
    # Prepare data
    (x_train, x_test, y_train, y_test), tokenizer, unique_labels = training_data(
        conf_data["dataset_path"],
        conf_data["text_column"],
        conf_data["label_column"],
        test_size=conf_train.get("test_set_size", 0.2),
        subset_size=-1,
        max_length=conf_data["max_words_per_message"],
        pad_to_max_length=conf_data.get("pad_to_max_length", True),
    )
    
    # Create and train model
    model = DistilBertClassifier(
        num_labels=y_train.shape[1],
        learning_rate=conf_train.get("learning_rate", 5e-5),
    )
    
    # Train the model with smaller batch size
    test_loss, test_accuracy = train_model(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        epochs=conf_train.get("epochs", 3),
        batch_size=conf_train.get("batch_size", 16),  # Use smaller batch size
        callbacks=define_callbacks(
            patience=conf_train.get("early_stopping_patience", 2),
            min_delta=conf_train.get("early_stopping_min_delta_acc", 0.01),
        ),
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and tokenizer
    save_model(model, tokenizer)
    
    # Save unique labels for prediction
    with open("models/unique_labels.txt", "w") as f:
        f.write("\n".join(unique_labels)) 