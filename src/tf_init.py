"""
Initialize TensorFlow environment correctly to avoid import conflicts with Transformers.
This should be imported before any other TensorFlow or Transformers imports.
"""
import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Try to initialize TensorFlow in a way that avoids RaggedTensorSpec conflicts
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Make sure keras is imported from tensorflow
    from tensorflow import keras
    
    # Use mixed precision for better performance on GPUs if available
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available, using mixed precision")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("No GPU detected")
    
    # Create initializers to avoid future import order issues
    initializers = {}
    
    # Force initialization of certain TensorFlow components
    initializers['ragged'] = tf.ragged.constant([[1, 2], [3]])
    
    print("TensorFlow initialized successfully.")
except Exception as e:
    print(f"Error initializing TensorFlow: {e}")
    sys.exit(1) 