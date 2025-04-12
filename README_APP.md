# IT Support Ticket Classifier App

This application classifies IT support tickets into three categories using a machine learning model trained on customer support data.

## Categories

The classifier categorizes support tickets into these three groups:

1. **Technical/IT Support** - Technical issues related to IT systems, networks, hardware, or software.
2. **Customer & Product Support** - Questions about products, usage, features, or general customer service inquiries.
3. **Financial/Other** - Billing, payments, returns, HR matters, or other administrative queries.

## Getting Started

### Prerequisites

- Python 3.6+
- Required packages: streamlit, pandas, scikit-learn, nltk

### Installation

1. Install required packages:
   ```
   pip install streamlit pandas scikit-learn nltk
   ```

2. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

### Training the Model

The model is trained using the CSV files in the `data/` directory:
- `ticket_train.csv`: Training dataset
- `ticket_valid.csv`: Validation dataset
- `ticket_test.csv`: Test dataset

To train the model, run:

```
python train_model.py
```

This will:
1. Load the datasets
2. Preprocess the text data
3. Train a model (Naive Bayes by default)
4. Evaluate the model on the test set
5. Save the trained model to `models/text_classifier.pkl`

### Running the App

To launch the Streamlit app:

```
streamlit run app.py
```

This will start a local web server and open the app in your default browser.

## Using the App

1. **Enter Text**: Type or paste a support ticket text in the input area.
2. **Try Examples**: Alternatively, select one of the sample texts from the dropdown menu and click "Use this sample".
3. **Classify**: Click the "Classify" button to see the prediction.
4. **View Results**: The predicted category and confidence scores will be displayed.
5. **Explore Data**: You can explore the training datasets by clicking "Load Datasets" in the sidebar.

## Customizing the Classifier

You can modify the training parameters in `train_model.py`:

- Change the classifier type between 'NB' (Naive Bayes) and 'SVM' (Support Vector Machine)
- Enable/disable grid search for hyperparameter tuning
- Adjust preprocessing steps (stopwords removal, stemming, etc.)

## Troubleshooting

- If the model file is missing, the app will create a simple demo model automatically
- Make sure your CSV files have the correct format and column names (text_en, queue)
- Check that your data folder contains all three required CSV files 