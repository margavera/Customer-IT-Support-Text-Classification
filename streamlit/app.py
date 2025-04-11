import streamlit as st
import tensorflow as tf
import numpy as np
import os
import sys
import pickle

# Add the parent directory to the path to import local modules
sys.path.append(os.path.abspath('.'))

# Import the required modules
from src.model import DistilBertClassifier, model_predict
from src.utils import encode_texts

# Set page configuration
st.set_page_config(
    page_title="IT Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'predicted_category' not in st.session_state:
    st.session_state.predicted_category = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'has_prediction' not in st.session_state:
    st.session_state.has_prediction = False

# Define the three consolidated categories
consolidated_categories = [
    "Technical/IT Support",
    "Customer & Product Support",
    "Financial/Other"
]

# Function to load the model and tokenizer
@st.cache_resource
def load_model_resources():
    """Load the saved model, tokenizer, and category labels"""
    try:
        # Path to the saved model
        model_folder = "models/distilbert_3class"
        
        # Check if the directory exists
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)
            st.warning(f"Model folder {model_folder} not found, created a new one")
            return None, None
        
        # Try to load category labels
        try:
            with open(f"{model_folder}/categories.txt", "r") as f:
                categories = f.read().splitlines()
        except FileNotFoundError:
            # If categories file doesn't exist, use the default categories
            categories = consolidated_categories
            with open(f"{model_folder}/categories.txt", "w") as f:
                f.write("\n".join(categories))
            
        # Try to load tokenizer
        try:
            with open(os.path.join(model_folder, "tokenizer.pkl"), "rb") as f:
                tokenizer = pickle.load(f)
        except FileNotFoundError:
            from transformers import DistilBertTokenizer
            # If tokenizer doesn't exist, use the default one
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
            # Save the tokenizer
            with open(os.path.join(model_folder, "tokenizer.pkl"), "wb") as f:
                pickle.dump(tokenizer, f)
            
        # Create the model (with 3 classes)
        model = DistilBertClassifier(
            num_labels=len(categories),
            learning_rate=3e-5,
            dropout_rate=0.2  # Use the same dropout as in the notebook to address overfitting
        )
        
        # Create a dummy input to build the model
        dummy_input = tf.ones((1, 128), dtype=tf.int32)
        dummy_mask = tf.ones((1, 128), dtype=tf.int32)
        model({"input_ids": dummy_input, "attention_mask": dummy_mask})
        
        # Try to load the model weights
        try:
            model.load_weights(os.path.join(model_folder, "model_weights"))
            print("Loaded model weights successfully")
        except:
            st.warning("No trained model weights found. The app will use an untrained model.")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to predict category
def predict_category(text, model, tokenizer):
    """Predict the category for a given text"""
    try:
        # Encode the text
        encoded_text = encode_texts(tokenizer, [text])
        
        # Use batched prediction to avoid memory issues
        logits = model.predict_in_batches(encoded_text, batch_size=1)[0]
        
        # Get probabilities
        probs = tf.nn.softmax(logits).numpy()
        
        # Get predicted class
        pred_idx = np.argmax(probs)
        
        return consolidated_categories[pred_idx], probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Main function
def main():
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #1E88E5;
    }
    .subtitle {
        text-align: center;
        color: #616161;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .prob-container {
        margin-top: 20px;
    }
    .prob-label {
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.markdown("<h1 class='title'>IT Support Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter a support ticket text to classify it into one of three categories</p>", unsafe_allow_html=True)
    
    # Load model resources
    model, tokenizer = load_model_resources()
    
    if not model or not tokenizer:
        st.error("Failed to load model resources. Please check the model folder.")
        return
    
    # Category descriptions
    category_descriptions = {
        "Technical/IT Support": "Technical issues related to IT systems, networks, hardware, or software",
        "Customer & Product Support": "Questions about products, usage, features, or general customer service inquiries",
        "Financial/Other": "Billing, payments, returns, HR matters, or other administrative queries"
    }
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Enter support ticket text:",
            height=200,
            placeholder="Example: I'm having trouble accessing my email account. The system keeps saying my password is incorrect even though I'm sure it's right."
        )
        
        # Sample texts
        st.markdown("#### Sample texts to try:")
        sample_texts = [
            "The server is down and we can't access our client database. This is urgent!",
            "I want to understand the features of your premium subscription plan.",
            "I'd like to request a refund for my recent purchase."
        ]
        
        # Create buttons for sample texts
        for i, sample in enumerate(sample_texts):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                text_input = sample
                st.session_state.text_input = sample
                st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Classify button
        if st.button("Classify", type="primary"):
            if not text_input:
                st.warning("Please enter some text to classify.")
            else:
                with st.spinner("Classifying..."):
                    # Get prediction
                    predicted_category, probabilities = predict_category(text_input, model, tokenizer)
                    
                    if predicted_category and probabilities is not None:
                        # Store results in session state
                        st.session_state.predicted_category = predicted_category
                        st.session_state.probabilities = probabilities
                        st.session_state.has_prediction = True
    
    with col2:
        st.markdown("### Categories")
        
        # Display category descriptions
        for category, description in category_descriptions.items():
            st.markdown(f"**{category}**")
            st.markdown(f"{description}")
            st.markdown("---")
    
    # If we have a prediction to show
    if st.session_state.has_prediction:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 class='prediction'>Prediction Results</h3>", unsafe_allow_html=True)
        
        predicted_category = st.session_state.predicted_category
        probabilities = st.session_state.probabilities
        
        # Create 3 columns for the results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Predicted Category: {predicted_category}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>{category_descriptions[predicted_category]}</p>", unsafe_allow_html=True)
            
            # Show probabilities
            st.markdown("<div class='prob-container'>", unsafe_allow_html=True)
            for i, category in enumerate(consolidated_categories):
                prob = probabilities[i] * 100
                st.markdown(f"<p class='prob-label'>{category}</p>", unsafe_allow_html=True)
                st.progress(float(prob / 100))
                st.markdown(f"<p style='text-align: right'>{prob:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()