import streamlit as st
import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add the parent directory to the path to import local modules
sys.path.append('..')

# Set page configuration
st.set_page_config(
    page_title="Ticket Classifier",
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
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

@st.cache_resource
def load_model():
    """Load the saved CNN model"""
    try:
        model = tf.keras.models.load_model('../models/model_cnn_glove.h5') 
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_category(text, model):
    """Predict the category for a given text"""
    try:
        preprocessed_text = preprocess_text(text)
        prediction = model.predict(np.array([preprocessed_text]))
        probs = tf.nn.softmax(prediction).numpy()
        
        # Get the predicted category
        pred_idx = np.argmax(probs)
        
        categories = ["Technical/IT Support", "Customer & Product Support", "Financial/Other"]
        return categories[pred_idx], probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def preprocess_text(text):
    """Preprocess text (tokenization and padding)"""
    tokenizer = Tokenizer(num_words=11000)  
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    
    padded_sequence = pad_sequences(sequences, maxlen=400)

    return padded_sequence.squeeze()

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
    
    st.markdown("<h1 class='title'>Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter a support ticket text to classify it into one of three categories</p>", unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if not model:
        st.error("Failed to load model. Please check the model path.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Enter support ticket text:",
            value=st.session_state.text_input,  
            height=200,
            placeholder="Example: I'm having trouble accessing my email account."
        )
        
        # Sample texts
        st.markdown("#### Sample texts to try:")
        sample_texts = [
            "The server is down and we can't access our client database.",
            "Data users have problems with delayed data psychronization process due to increased server load. Although optimizations have already been carried out, there are still difficulties.",
            "I recently observed an unanticipated charge on my monthly subscription bill and suspect it might be a result of an incorrect adjustment to my subscription plan."
        ]
        
        # Create buttons for sample texts
        for i, sample in enumerate(sample_texts):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                st.session_state.text_input = sample 
                st.rerun() 
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Classify button
        if st.button("Classify", type="primary"):
            if not text_input:
                st.warning("Please enter some text to classify.")
            else:
                with st.spinner("Classifying..."):
                    # Get prediction
                    predicted_category, probabilities = predict_category(text_input, model)
                    
                    if predicted_category and probabilities is not None:
                        # Store results in session state
                        st.session_state.predicted_category = predicted_category
                        st.session_state.probabilities = probabilities
                        st.session_state.has_prediction = True
    
    with col2:
        st.markdown("### Categories")
        
        # Display category descriptions
        category_descriptions = {
            "Technical/IT Support": "Technical issues related to IT systems, networks, hardware, or software",
            "Customer & Product Support": "Questions about products, usage, features, or general customer service inquiries",
            "Financial/Other": "Billing, payments, returns, HR matters, or other administrative queries"
        }

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

            # Show probabilities
            st.markdown("<div class='prob-container'>", unsafe_allow_html=True)
            
            categories = ["Technical/IT Support", "Customer & Product Support", "Financial/Other"]

            for i, category in enumerate(categories):
                prob = probabilities[0][i]  
                
                st.markdown(f"<p class='prob-label'>{category}</p>", unsafe_allow_html=True)
                st.progress(float(prob)) 
                st.markdown(f"<p style='text-align: right'>{prob * 100:.2f}%</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
