# Text Classification for Customer Support Tickets

This project implements a text classification system for customer support tickets using traditional NLP techniques with scikit-learn. It provides tools for training, evaluating, and deploying machine learning models to categorize support tickets based on their content.

## Project Structure

```
.
├── data/                  # Dataset files
├── models/                # Trained models and model artifacts
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── TextClassification.ipynb  # Model training and evaluation
├── src/                   # Source code
│   ├── config.py          # Configuration settings
│   ├── evaluate.py        # Model evaluation tools
│   ├── models.py          # Model definitions
│   ├── preprocessing.py   # Text preprocessing functions
│   └── utils.py           # Utility functions
└── README.md              # This file
```

## Features

- **Text Processing**: Tokenization, stemming, stopword removal, and TF-IDF vectorization.
- **Multiple Models**: Support for Naive Bayes and SVM classifiers.
- **Hyperparameter Tuning**: Grid search for optimal model parameters.
- **Evaluation**: Comprehensive evaluation metrics including confusion matrices, classification reports, and visualizations.
- **Model Persistence**: Save and load trained models for reuse.
- **Exploratory Analysis**: Tools for understanding data distribution and text characteristics.

## Requirements

- Python 3.6+
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- wordcloud (for EDA notebook)

## Getting Started

1. **Clone the repository**

2. **Install dependencies**
   ```
   pip install scikit-learn pandas numpy nltk matplotlib seaborn wordcloud
   ```

3. **Prepare your data**
   - Place your CSV dataset in the `data/` directory
   - Format: CSV file with a column for text content and a column for labels

4. **Run exploratory data analysis**
   - Open `notebooks/EDA.ipynb` in Jupyter
   - Modify the file path and column names as needed
   - Execute the notebook to understand your data

5. **Train and evaluate models**
   - Open `notebooks/TextClassification.ipynb` in Jupyter
   - Adjust the configuration settings for your dataset
   - Execute the notebook to train and evaluate models

## Using the Source Code

### Configuration

Modify `src/config.py` to set default parameters for data loading, preprocessing, training, and evaluation.

```python
# Example: Updating configuration
from src.config import get_config, update_config

config = get_config()
config = update_config(config, 
    data={
        'dataset_path': 'data/my_tickets.csv',
        'text_column': 'content',
        'label_column': 'category'
    },
    training={
        'classifier': 'SVM',
        'use_grid_search': True
    }
)
```

### Preprocessing Text

```python
from src.preprocessing import preprocess_text, create_vectorizer

# Preprocess a single text
processed_text = preprocess_text("I need help with my account login")

# Create a vectorizer
vectorizer = create_vectorizer(remove_stop_words=True, use_stemming=False)
```

### Training Models

```python
from src.models import create_naive_bayes_pipeline, train_model

# Create a pipeline
pipeline = create_naive_bayes_pipeline(vectorizer, tfidf_transformer)

# Train the model
trained_model = train_model(pipeline, train_data, train_labels)
```

### Evaluating Models

```python
from src.evaluate import evaluate_model, plot_confusion_matrix

# Get predictions
predictions = trained_model.predict(test_data)

# Evaluate
accuracy = evaluate_model(predictions, test_labels)

# Visualize
plot_confusion_matrix(test_labels, predictions)
```

## Extending the Project

### Adding New Models

Extend `src/models.py` to add new classifier types:

```python
def create_random_forest_pipeline(vectorizer, tfidf_transformer):
    """Create a pipeline with Random Forest classifier"""
    from sklearn.ensemble import RandomForestClassifier
    
    return Pipeline([
        ('vect', vectorizer),
        ('tfidf', tfidf_transformer),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
```

### Customizing Preprocessing

Modify `src/preprocessing.py` to add custom preprocessing steps:

```python
def custom_preprocess(text):
    """Custom preprocessing for specific domain knowledge"""
    # Your custom preprocessing logic
    return processed_text
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Inspiration from the notebook template shared in the request
* scikit-learn documentation for best practices in text classification
* NLTK documentation for text processing techniques
