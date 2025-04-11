import os
from typing import List, Any, Union, Tuple, Sequence

import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

# Import with explicit try-except to handle TensorFlow conflicts
try:
    from transformers import TFDistilBertModel, DistilBertConfig, DistilBertTokenizer
except RuntimeError as e:
    print(f"Warning: {e}")
    print("Trying alternative import method...")
    import transformers
    TFDistilBertModel = transformers.TFDistilBertModel
    DistilBertConfig = transformers.DistilBertConfig
    DistilBertTokenizer = transformers.DistilBertTokenizer

from src.utils import encode_texts


class DistilBertClassifier(keras.Model):
    def __init__(
        self,
        num_labels: int,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.2,
        metrics: List[str] = ["accuracy"],
    ):
        super(DistilBertClassifier, self).__init__()
        
        # Create DistilBERT model directly instead of using TFDistilBertForSequenceClassification
        try:
            config = DistilBertConfig.from_pretrained("distilbert-base-cased")
            self.distilbert = TFDistilBertModel.from_pretrained(
                "distilbert-base-cased", config=config
            )
            distil_classifier_out_dim = config.dim
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Using random initialization instead...")
            config = DistilBertConfig()
            self.distilbert = TFDistilBertModel(config)
            distil_classifier_out_dim = config.dim

        self.dense1 = keras.layers.Dense(
            distil_classifier_out_dim, activation="relu", name="dense1"
        )
        self.dense2 = keras.layers.Dense(
            distil_classifier_out_dim // 2, activation="relu", name="dense2"
        )
        self.dense3 = keras.layers.Dense(
            distil_classifier_out_dim // 4, activation="relu", name="dense3"
        )
        self.dense4 = keras.layers.Dense(num_labels, name="dense4")
        self.dropout = keras.layers.Dropout(dropout_rate)

        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.compile(
            loss=loss_fn, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=metrics
        )

    def call(
        self, inputs: Union[dict, tf.Tensor], training=None, **kwargs: Any
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, None]]:
        # Handle both dictionary input and tensor input
        if isinstance(inputs, dict):
            distilbert_output = self.distilbert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                training=training,
                **kwargs
            )
        else:
            distilbert_output = self.distilbert(inputs, training=training, **kwargs)

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dense1(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        pooled_output = self.dense3(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.dense4(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        return outputs
    
    def predict_in_batches(self, inputs: dict, batch_size: int = 16) -> np.ndarray:
        """
        Makes predictions in small batches to avoid memory issues.
        
        Args:
            inputs: Dictionary with 'input_ids' and 'attention_mask'
            batch_size: Size of batches to process
            
        Returns:
            NumPy array of predictions
        """
        # Get total number of samples
        num_samples = len(inputs["input_ids"])
        
        # Initialize output array
        all_logits = []
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_inputs = {
                "input_ids": inputs["input_ids"][i:end_idx],
                "attention_mask": inputs["attention_mask"][i:end_idx]
            }
            
            # Get predictions for this batch
            batch_logits = self(batch_inputs, training=False)[0].numpy()
            all_logits.append(batch_logits)
            
            # Clear GPU memory
            tf.keras.backend.clear_session()
            
        # Concatenate results
        return np.vstack(all_logits)


def save_model(
    model: DistilBertClassifier, tokenizer: DistilBertTokenizer, model_folder="models"
) -> None:
    """
    Saves the model and tokenizer to disk.
    
    Args:
        model: The DistilBERT classifier model to save
        tokenizer: The tokenizer to save
        model_folder: Directory where to save the model
    """
    os.makedirs(model_folder, exist_ok=True)
    model.save_weights(os.path.join(model_folder, "model_weights"))
    with open(os.path.join(model_folder, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)


def load_model(
    model_folder: str = "models",
    num_labels: int = 10,
    learning_rate: float = 5e-5
) -> Tuple[DistilBertClassifier, DistilBertTokenizer]:
    """
    Loads a saved model and tokenizer.
    
    Args:
        model_folder: Directory where the model is saved
        num_labels: Number of classification labels
        learning_rate: Learning rate to use when creating the model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = DistilBertClassifier(num_labels=num_labels, learning_rate=learning_rate)
    # Create a dummy input to build the model
    dummy_input = tf.ones((1, 128), dtype=tf.int32)
    dummy_mask = tf.ones((1, 128), dtype=tf.int32)
    model({"input_ids": dummy_input, "attention_mask": dummy_mask})
    
    model.load_weights(os.path.join(model_folder, "model_weights"))
    
    with open(os.path.join(model_folder, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer


def model_predict(
    model: DistilBertClassifier, tokenizer: DistilBertTokenizer, texts: Sequence[str], batch_size: int = 16
) -> List[int]:
    """
    Makes predictions using the model.
    
    Args:
        model: The DistilBERT classifier model
        tokenizer: The tokenizer
        texts: List of texts to classify
        batch_size: Size of batches to process to avoid memory issues
        
    Returns:
        List of predicted class indices
    """
    encoded_inputs = encode_texts(tokenizer, texts)
    
    # Use batched prediction to avoid memory issues
    predictions = model.predict_in_batches(encoded_inputs, batch_size=batch_size)
    
    return predictions.argmax(axis=1).tolist() 