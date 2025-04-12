import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from typing import Dict, List, Tuple, Union, Any
import datetime

# Import transformers with error handling for compatibility
try:
    from transformers import TFDistilBertModel, DistilBertConfig, DistilBertTokenizer
except ImportError:
    try:
        from transformers.models.distilbert.modeling_tf_distilbert import TFDistilBertModel
        from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
        from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
    except ImportError:
        print("Warning: Could not import DistilBERT classes from transformers library.")
        TFDistilBertModel, DistilBertConfig, DistilBertTokenizer = None, None, None

#
# TRADITIONAL ML MODELS 
#

def create_naive_bayes_pipeline(vectorizer=None, tfidf_transformer=None, fit_prior=True):
    """
    Create a pipeline with Naive Bayes classifier
    
    Parameters:
    -----------
    vectorizer : CountVectorizer, default=None
        Vectorizer instance. If None, a default one will be created.
    tfidf_transformer : TfidfTransformer, default=None
        TF-IDF transformer instance. If None, a default one will be created.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities in the Naive Bayes model.
        
    Returns:
    --------
    pipeline : Pipeline
        Sklearn pipeline with Naive Bayes classifier
    """
    if vectorizer is None:
        vectorizer = CountVectorizer(stop_words='english')
    
    if tfidf_transformer is None:
        tfidf_transformer = TfidfTransformer()
    
    return Pipeline([
        ('vect', vectorizer),
        ('tfidf', tfidf_transformer),
        ('clf', MultinomialNB(fit_prior=fit_prior))
    ])

def create_svm_pipeline(vectorizer=None, tfidf_transformer=None, C=1.0, class_weight=None):
    """
    Create a pipeline with SVM classifier
    
    Parameters:
    -----------
    vectorizer : CountVectorizer, default=None
        Vectorizer instance. If None, a default one will be created.
    tfidf_transformer : TfidfTransformer, default=None
        TF-IDF transformer instance. If None, a default one will be created.
    C : float, default=1.0
        Regularization parameter for SVM.
    class_weight : dict or 'balanced', default=None
        Class weights for SVM.
        
    Returns:
    --------
    pipeline : Pipeline
        Sklearn pipeline with SVM classifier
    """
    if vectorizer is None:
        vectorizer = CountVectorizer(stop_words='english')
    
    if tfidf_transformer is None:
        tfidf_transformer = TfidfTransformer()
    
    return Pipeline([
        ('vect', vectorizer),
        ('tfidf', tfidf_transformer),
        ('clf', LinearSVC(C=C, class_weight=class_weight, random_state=42))
    ])

def get_grid_search_params(classifier_type='NB'):
    """
    Get parameters for grid search based on classifier type
    
    Parameters:
    -----------
    classifier_type : str, default='NB'
        Type of classifier ('NB' or 'SVM')
        
    Returns:
    --------
    params : dict
        Parameters for grid search
    """
    if classifier_type == 'NB':
        return {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
        }
    elif classifier_type == 'SVM':
        return {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__C': (0.1, 1, 10),
            'clf__class_weight': (None, 'balanced')
        }
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def create_vectorizer(remove_stop_words=True, ngram_range=(1, 2), max_features=10000, min_df=2):
    """
    Create a CountVectorizer with specified parameters
    
    Parameters:
    -----------
    remove_stop_words : bool, default=True
        Whether to remove stop words
    ngram_range : tuple, default=(1, 2)
        The lower and upper boundary of the range of n-values for n-grams to be extracted
    max_features : int, default=10000
        Maximum number of features (vocabularies) to extract
    min_df : int, default=2
        Minimum document frequency (ignore terms that appear in fewer than min_df documents)
        
    Returns:
    --------
    vectorizer : CountVectorizer
        Initialized CountVectorizer
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Set stop words based on parameter
    stop_words = 'english' if remove_stop_words else None
    
    # Create vectorizer
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        token_pattern=r'\b\w+\b'  # Match any word character
    )
    
    return vectorizer

def create_tfidf_transformer(use_idf=True, norm='l2', smooth_idf=True):
    """
    Create a TF-IDF transformer with specified parameters
    
    Parameters:
    -----------
    use_idf : bool, default=True
        Whether to use inverse document frequency
    norm : str, default='l2'
        Normalization method ('l1', 'l2', or None)
    smooth_idf : bool, default=True
        Whether to smooth IDF weights by adding 1 to document frequencies
        
    Returns:
    --------
    transformer : TfidfTransformer
        Initialized TF-IDF transformer
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    
    # Create TF-IDF transformer
    transformer = TfidfTransformer(
        use_idf=use_idf,
        norm=norm,
        smooth_idf=smooth_idf
    )
    
    return transformer

def create_grid_search(pipeline, params, n_jobs=-1, cv=5):
    """
    Create a grid search instance
    
    Parameters:
    -----------
    pipeline : Pipeline
        Sklearn pipeline
    params : dict
        Parameters for grid search
    n_jobs : int, default=-1
        Number of parallel jobs
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    grid_search : GridSearchCV
        Grid search instance
    """
    return GridSearchCV(pipeline, params, n_jobs=n_jobs, cv=cv, verbose=1)

def train_ml_model(model, train_data, train_labels, validation_data=None, validation_labels=None):
    """
    Train a model
    
    Parameters:
    -----------
    model : Pipeline or GridSearchCV
        Model to train
    train_data : array-like
        Training data
    train_labels : array-like
        Training labels
    validation_data : array-like, optional
        Validation data
    validation_labels : array-like, optional
        Validation labels
        
    Returns:
    --------
    model : Pipeline or GridSearchCV
        Trained model
    """
    print("Training model...")
    start_time = time.time()
    
    model.fit(train_data, train_labels)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    if isinstance(model, GridSearchCV):
        print(f"Best score: {model.best_score_:.4f}")
        print(f"Best parameters: {model.best_params_}")
        model = model.best_estimator_
    
    if validation_data is not None and validation_labels is not None:
        val_score = model.score(validation_data, validation_labels)
        print(f"Validation score: {val_score:.4f}")
    
    return model

def predict(model, test_data):
    """
    Make predictions using a trained model
    
    Parameters:
    -----------
    model : Pipeline or GridSearchCV
        Trained model
    test_data : array-like
        Test data
        
    Returns:
    --------
    predictions : array-like
        Predicted labels
    """
    print("Making predictions...")
    if isinstance(model, GridSearchCV):
        return model.best_estimator_.predict(test_data)
    else:
        return model.predict(test_data)

def evaluate_traditional_model(model, test_data, test_labels):
    """
    Evaluate a traditional ML model (scikit-learn)
    
    Parameters:
    -----------
    model : Pipeline or GridSearchCV
        Trained model
    test_data : array-like
        Test data
    test_labels : array-like
        True labels
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    predictions = predict(model, test_data)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    
    # Generate classification report
    report = classification_report(test_labels, predictions, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))
    print("\nConfusion Matrix:")
    print(cm)
    
    return metrics

#
# DEEP LEARNING MODELS
#

class DistilBertClassifier(tf.keras.Model):
    """
    DistilBERT model for text classification
    """
    def __init__(self, num_labels=3, learning_rate=3e-5, dropout_rate=0.2, metrics=None):
        """
        Initialize the DistilBERT classifier
        
        Parameters:
        -----------
        num_labels : int, default=3
            Number of output classes
        learning_rate : float, default=3e-5
            Learning rate for the optimizer
        dropout_rate : float, default=0.2
            Dropout rate for regularization
        metrics : list, default=None
            List of metrics to track
        """
        super(DistilBertClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # Initialize DistilBERT
        try:
            self.config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=self.config)
        except:
            print("Error loading pretrained DistilBERT model. Using random initialization.")
            self.config = DistilBertConfig()
            self.distilbert = TFDistilBertModel(self.config)
        
        # Define classification layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_labels, 
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                              name="classifier")
        
        # Compile model
        self.compile_model(metrics)
    
    def call(self, inputs, training=False):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : dict
            Input tensors containing 'input_ids' and 'attention_mask'
        training : bool, default=False
            Whether in training mode
            
        Returns:
        --------
        logits : tensor
            Output logits
        """
        # Get DistilBERT outputs
        distilbert_output = self.distilbert(inputs)
        hidden_state = distilbert_output[0]  # Get the sequence output
        pooled_output = hidden_state[:, 0]   # Get the CLS token output
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def compile_model(self, metrics=None):
        """
        Compile the model
        
        Parameters:
        -----------
        metrics : list, default=None
            List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=metrics
        )
    
    def predict_in_batches(self, inputs, batch_size=8):
        """
        Make predictions in batches to avoid OOM errors
        
        Parameters:
        -----------
        inputs : dict
            Input tensors containing 'input_ids' and 'attention_mask'
        batch_size : int, default=8
            Batch size for predictions
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        # Get total number of samples
        n_samples = inputs['input_ids'].shape[0]
        
        # Initialize predictions array
        all_logits = []
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            # Get batch
            batch = {
                'input_ids': inputs['input_ids'][i:i+batch_size],
                'attention_mask': inputs['attention_mask'][i:i+batch_size]
            }
            
            # Get predictions
            logits = self(batch, training=False)
            all_logits.append(logits.numpy())
        
        # Concatenate predictions
        return np.vstack(all_logits)

def encode_texts(tokenizer, texts, max_length=128):
    """
    Encode texts using the tokenizer
    
    Parameters:
    -----------
    tokenizer : DistilBertTokenizer
        Tokenizer to use
    texts : list
        List of texts to encode
    max_length : int, default=128
        Maximum sequence length
        
    Returns:
    --------
    encodings : dict
        Dictionary with 'input_ids' and 'attention_mask'
    """
    # Tokenize the texts
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    
    return encodings

def encode_labels(labels, unique_labels=None):
    """
    Encode labels as one-hot vectors
    
    Parameters:
    -----------
    labels : list
        List of labels to encode
    unique_labels : list, default=None
        List of unique labels. If None, it will be inferred from the data.
        
    Returns:
    --------
    encoded_labels : np.ndarray
        One-hot encoded labels
    """
    if unique_labels is None:
        unique_labels = sorted(list(set(labels)))
    
    # Create label mapping
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    label_indices = [label_map[label] for label in labels]
    
    # Convert to one-hot vectors
    encoded_labels = tf.keras.utils.to_categorical(label_indices, num_classes=len(unique_labels))
    
    return encoded_labels

def save_model(model, model_path, tokenizer=None, tokenizer_path=None):
    """
    Save a model and optionally its tokenizer
    
    Parameters:
    -----------
    model : tf.keras.Model or Pipeline
        Model to save
    model_path : str
        Path to save the model
    tokenizer : Tokenizer, default=None
        Tokenizer to save
    tokenizer_path : str, default=None
        Path to save the tokenizer
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save based on model type
    if isinstance(model, tf.keras.Model):
        # Save TensorFlow model
        model.save_weights(model_path)
        print(f"TensorFlow model saved to {model_path}")
    else:
        # Save scikit-learn model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Scikit-learn model saved to {model_path}")
    
    # Save tokenizer if provided
    if tokenizer is not None and tokenizer_path is not None:
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")

def load_model(model_path, model_type='sklearn', num_labels=3):
    """
    Load a saved model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    model_type : str, default='sklearn'
        Type of model ('sklearn' or 'tensorflow')
    num_labels : int, default=3
        Number of labels (only used for TensorFlow models)
        
    Returns:
    --------
    model : Model
        Loaded model
    """
    if model_type == 'sklearn':
        # Load scikit-learn model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Scikit-learn model loaded from {model_path}")
    else:
        # Create and load TensorFlow model
        model = DistilBertClassifier(num_labels=num_labels)
        
        # Create dummy input to build the model
        dummy_input = {
            'input_ids': tf.ones((1, 128), dtype=tf.int32),
            'attention_mask': tf.ones((1, 128), dtype=tf.int32)
        }
        _ = model(dummy_input)
        
        # Load weights
        model.load_weights(model_path)
        print(f"TensorFlow model loaded from {model_path}")
    
    return model

def model_predict(model, tokenizer, texts, batch_size=8):
    """
    Make predictions with a DistilBERT model
    
    Parameters:
    -----------
    model : DistilBertClassifier
        Trained model
    tokenizer : DistilBertTokenizer
        Tokenizer
    texts : list
        List of texts to classify
    batch_size : int, default=8
        Batch size for predictions
        
    Returns:
    --------
    predictions : list
        Predicted class indices
    """
    # Encode texts
    encoded_texts = encode_texts(tokenizer, texts)
    
    # Get predictions
    logits = model.predict_in_batches(encoded_texts, batch_size=batch_size)
    
    # Get predicted class indices
    predictions = np.argmax(logits, axis=1).tolist()
    
    return predictions

def create_distilbert_model(num_labels=3, dropout_rate=0.15):
    """
    Create a DistilBERT model using a simpler approach that avoids compatibility issues
    
    Parameters:
    -----------
    num_labels : int, default=3
        Number of output classes
    dropout_rate : float, default=0.15
        Dropout rate for regularization
        
    Returns:
    --------
    model : tf.keras.Model
        Ready-to-use DistilBERT model
    """
    # Create a complete model using model subclassing instead of functional API
    class CompatibleDistilBertModel(tf.keras.Model):
        def __init__(self, num_labels, dropout_rate):
            super().__init__()
            
            # Load DistilBERT without initialization issues
            try:
                self.config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
                self.bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased', 
                                                           config=self.config)
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Using random initialization...")
                self.config = DistilBertConfig()
                self.bert = TFDistilBertModel(self.config)
            
            # Classification layers
            self.dense1 = tf.keras.layers.Dense(256, activation='gelu')
            self.layer_norm = tf.keras.layers.LayerNormalization()
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dense2 = tf.keras.layers.Dense(64, activation='gelu')
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate/2)
            self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')
        
        def call(self, inputs, training=False):
            # Handle both dictionary and list/tuple inputs
            if isinstance(inputs, dict):
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            else:
                input_ids = inputs[0]
                attention_mask = inputs[1]
            
            # Get DistilBERT outputs - use low-level TF ops to avoid Keras issues
            bert_outputs = self.bert(
                input_ids=tf.convert_to_tensor(input_ids),
                attention_mask=tf.convert_to_tensor(attention_mask),
                training=training
            )
            
            # Get the [CLS] token embedding (first token)
            cls_output = bert_outputs[0][:, 0, :]
            
            # Apply classification layers
            x = self.dense1(cls_output)
            x = self.layer_norm(x)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            outputs = self.classifier(x)
            
            return outputs
        
        def build_model(self):
            """Build the model with dummy inputs to initialize weights"""
            input_shape = (1, 128)  # Batch size 1, sequence length 128
            dummy_input_ids = tf.ones(input_shape, dtype=tf.int32)
            dummy_attention_mask = tf.ones(input_shape, dtype=tf.int32)
            _ = self([dummy_input_ids, dummy_attention_mask], training=False)
            
    # Initialize the model
    model = CompatibleDistilBertModel(num_labels, dropout_rate)
    
    # Build the model to initialize all layers
    model.build_model()
    
    return model

def weighted_categorical_crossentropy(class_weights):
    """
    Create a weighted categorical crossentropy loss function
    
    Parameters:
    -----------
    class_weights : dict or numpy array
        Weights for each class, indexed by class number
        
    Returns:
    --------
    loss_fn : function
        Weighted loss function that can be used in model.compile()
    """
    def loss(y_true, y_pred):
        # Convert class_weights to tensor if it's a dict
        if isinstance(class_weights, dict):
            weights_tensor = tf.constant([class_weights[i] for i in range(len(class_weights))], 
                                        dtype=tf.float32)
        else:
            weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        
        # Standard categorical crossentropy
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        
        # Get class indices from one-hot encoded true labels
        class_indices = tf.argmax(y_true, axis=1)
        
        # Get weights for each sample based on its true class
        sample_weights = tf.gather(weights_tensor, class_indices)
        
        # Apply weights to the loss
        weighted_loss = cce * sample_weights
        
        # Return mean loss
        return tf.reduce_mean(weighted_loss)
    
    return loss

def train_dl_model(model, x_train, x_valid, y_train, y_valid, epochs=10, batch_size=16, 
                 eval_batch_size=32, callbacks=None):
    """
    Train a deep learning model and return its final validation metrics
    
    Parameters:
    -----------
    model : tf.keras.Model
        The model to train
    x_train : dict, list, or BatchEncoding
        Training inputs (either a dict with 'input_ids' and 'attention_mask', 
        a list of these tensors, or a BatchEncoding object)
    x_valid : dict, list, or BatchEncoding
        Validation inputs
    y_train : array-like
        Training labels
    y_valid : array-like
        Validation labels
    epochs : int, default=10
        Number of training epochs
    batch_size : int, default=16
        Batch size for training
    eval_batch_size : int, default=32
        Batch size for evaluation
    callbacks : list, default=None
        List of Keras callbacks
        
    Returns:
    --------
    val_loss : float
        Final validation loss
    val_accuracy : float
        Final validation accuracy
    """
    print(f"Training with batch size: {batch_size}, evaluation batch size: {eval_batch_size}")
    print(f"Training on {len(y_train)} samples, validating on {len(y_valid)} samples")
    
    # Convert BatchEncoding to dict if needed
    def prepare_inputs(inputs):
        # Check if it's a BatchEncoding object from transformers
        if hasattr(inputs, 'data') and isinstance(inputs.data, dict):
            # Convert BatchEncoding to dict of tensors
            return {
                'input_ids': tf.convert_to_tensor(inputs['input_ids']),
                'attention_mask': tf.convert_to_tensor(inputs['attention_mask'])
            }
        # For regular dict with tensors that aren't TF tensors
        elif isinstance(inputs, dict):
            return {
                'input_ids': tf.convert_to_tensor(inputs['input_ids']),
                'attention_mask': tf.convert_to_tensor(inputs['attention_mask'])
            }
        # If it's already a list or tuple of tensors
        return inputs
    
    # Prepare inputs
    x_train_prepared = prepare_inputs(x_train)
    x_valid_prepared = prepare_inputs(x_valid)
    
    # Train the model
    history = model.fit(
        x_train_prepared, 
        y_train,
        validation_data=(x_valid_prepared, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        validation_batch_size=eval_batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get the best validation metrics
    if callbacks and any(isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks):
        # If using early stopping with restore_best_weights=True
        val_loss = min(history.history['val_loss'])
        best_epoch_idx = history.history['val_loss'].index(val_loss)
        val_accuracy = history.history['val_accuracy'][best_epoch_idx]
    else:
        # Just use the last epoch
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
    
    print("Evaluating model in batches...")
    # Evaluate model on validation set
    val_metrics = model.evaluate(
        x_valid_prepared, 
        y_valid, 
        batch_size=eval_batch_size, 
        verbose=0
    )
    val_loss, val_accuracy = val_metrics[0], val_metrics[1]
    
    return val_loss, val_accuracy

def train_with_huggingface_trainer(
    tokenized_train_dataset, 
    tokenized_valid_dataset, 
    num_labels=3, 
    model_name="distilbert-base-uncased",
    epochs=3, 
    train_batch_size=16, 
    eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    output_dir="results"
):
    """
    Train a DistilBERT model using Hugging Face's TFTrainer
    
    Parameters:
    -----------
    tokenized_train_dataset : TFDataset or dict
        Tokenized training dataset in the format expected by Hugging Face
    tokenized_valid_dataset : TFDataset or dict
        Tokenized validation dataset
    num_labels : int, default=3
        Number of output classes
    model_name : str, default="distilbert-base-uncased"
        Pre-trained model name from Hugging Face
    epochs : int, default=3
        Number of training epochs
    train_batch_size : int, default=16
        Batch size for training
    eval_batch_size : int, default=32
        Batch size for evaluation
    warmup_steps : int, default=500
        Number of warmup steps for learning rate scheduler
    weight_decay : float, default=0.01
        Weight decay for regularization
    learning_rate : float, default=5e-5
        Learning rate
    output_dir : str, default="results"
        Directory to save model outputs
        
    Returns:
    --------
    model : TFDistilBertForSequenceClassification
        Trained model
    trainer : TFTrainer
        The trainer object that can be used for additional evaluation
    """
    # Import necessary Hugging Face components
    try:
        from transformers import TFTrainingArguments, TFTrainer, TFDistilBertForSequenceClassification
    except ImportError:
        raise ImportError(
            "Hugging Face transformers library is required. "
            "Please install it with: pip install transformers"
        )
    
    # Set up training arguments
    training_args = TFTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        save_steps=1000,
        evaluation_strategy="steps"
    )
    
    # Create model within the distribution strategy scope
    print(f"Initializing model with {num_labels} output classes...")
    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    
    # Create TFTrainer
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset
    )
    
    # Train the model
    print("Training with Hugging Face TFTrainer...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model, trainer

def prepare_datasets_for_trainer(x_train, y_train, x_valid, y_valid):
    """
    Prepare datasets for use with Hugging Face's TFTrainer
    
    Parameters:
    -----------
    x_train : dict or BatchEncoding
        Training inputs with 'input_ids' and 'attention_mask'
    y_train : array-like
        One-hot encoded training labels
    x_valid : dict or BatchEncoding
        Validation inputs
    y_valid : array-like
        One-hot encoded validation labels
        
    Returns:
    --------
    train_dataset : tf.data.Dataset
        Training dataset in format for TFTrainer
    valid_dataset : tf.data.Dataset
        Validation dataset in format for TFTrainer
    """
    import numpy as np
    
    # Convert y labels from one-hot to indices
    y_train_indices = np.argmax(y_train, axis=1)
    y_valid_indices = np.argmax(y_valid, axis=1)
    
    # Prepare train features
    train_features = {
        'input_ids': x_train['input_ids'].numpy() if hasattr(x_train['input_ids'], 'numpy') else x_train['input_ids'],
        'attention_mask': x_train['attention_mask'].numpy() if hasattr(x_train['attention_mask'], 'numpy') else x_train['attention_mask'],
        'labels': y_train_indices
    }
    
    # Prepare validation features
    valid_features = {
        'input_ids': x_valid['input_ids'].numpy() if hasattr(x_valid['input_ids'], 'numpy') else x_valid['input_ids'],
        'attention_mask': x_valid['attention_mask'].numpy() if hasattr(x_valid['attention_mask'], 'numpy') else x_valid['attention_mask'],
        'labels': y_valid_indices
    }
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_features)
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_features)
    
    return train_dataset, valid_dataset

def define_callbacks(patience=2, min_delta=0.01, checkpoint_path=None):
    """
    Define callbacks for model training
    
    Parameters:
    -----------
    patience : int, default=2
        Number of epochs with no improvement after which training will be stopped
    min_delta : float, default=0.01
        Minimum change in the monitored quantity to qualify as an improvement
    checkpoint_path : str, default=None
        Path to save model checkpoints. If None, model checkpoints won't be saved.
        
    Returns:
    --------
    callbacks : list
        List of Keras callbacks
    """
    callbacks = []
    
    # Add early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Add model checkpoint callback if path is provided
    if checkpoint_path:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    # Add TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks

def train_model(model, x_train, x_valid, y_train, y_valid, epochs=3, batch_size=32, 
              eval_batch_size=None, callbacks=None):
    """
    Train a model
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to train
    x_train : dict, tf.data.Dataset, or array-like
        Training data features
    x_valid : dict, tf.data.Dataset, or array-like
        Validation data features
    y_train : array-like
        Training data labels
    y_valid : array-like
        Validation data labels
    epochs : int, default=3
        Number of epochs to train the model
    batch_size : int, default=32
        Batch size for training
    eval_batch_size : int, default=None
        Batch size for evaluation. If None, uses the same as batch_size.
    callbacks : list, default=None
        List of Keras callbacks
        
    Returns:
    --------
    val_loss : float
        Validation loss
    val_accuracy : float
        Validation accuracy
    history : tf.keras.callbacks.History
        Training history object returned by model.fit()
    """
    import time
    import datetime
    import numpy as np
    
    # Set evaluation batch size if not provided
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    print(f"Training with batch size: {batch_size}, evaluation batch size: {eval_batch_size}")
    print(f"Training on {len(y_train)} samples, validating on {len(y_valid)} samples")
    
    # Prepare callbacks if not provided
    if callbacks is None:
        callbacks = define_callbacks()
    
    # Convert BatchEncoding to dict if needed (for transformers library)
    def prepare_inputs(inputs):
        # Check if it's a BatchEncoding object from transformers
        if hasattr(inputs, 'data') and isinstance(inputs.data, dict):
            # Convert BatchEncoding to dict of tensors
            return {
                'input_ids': tf.convert_to_tensor(inputs['input_ids']),
                'attention_mask': tf.convert_to_tensor(inputs['attention_mask'])
            }
        # For regular dict with tensors that aren't TF tensors
        elif isinstance(inputs, dict):
            return {
                'input_ids': tf.convert_to_tensor(inputs['input_ids']),
                'attention_mask': tf.convert_to_tensor(inputs['attention_mask'])
            }
        # If it's already a list or tuple of tensors
        return inputs
    
    # Prepare inputs for BatchEncoding objects
    if hasattr(x_train, 'data') or isinstance(x_train, dict):
        x_train = prepare_inputs(x_train)
        x_valid = prepare_inputs(x_valid)
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        validation_batch_size=eval_batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Find the epoch with the best validation accuracy
    best_epoch_idx = np.argmax(history.history['val_accuracy'])
    val_loss = history.history['val_loss'][best_epoch_idx]
    val_accuracy = history.history['val_accuracy'][best_epoch_idx]
    
    print(f"Best validation accuracy: {val_accuracy:.4f} (loss: {val_loss:.4f}) at epoch {best_epoch_idx + 1}")
    
    # Return history along with other metrics
    return val_loss, val_accuracy, history 