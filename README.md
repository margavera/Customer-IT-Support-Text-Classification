# Text Classification for Customer Support Tickets

Team members:

| Name                           | Email                               |
| -----------------------        | ----------------------------------- |
| Margarita Vera Cabrer          | marga.vera@alu.icai.comillas.edu    |
| Elena Martínez Torrijos        | 202407060@alu.comillas.edu          |
| Claudia Hermández de la Calera | chdelacalera@alu.comillas.edu       |


The dataset used in this project was obtained from Kaggle and is available at the following link:     
[Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?select=dataset-tickets-multi-lang3-4k.csv)   
 
It contains a collection of real-world customer support tickets written in english or german, along with metadata such as ticket subject, body, language, assigned queue, priority, and various tags.    
 
We chose this dataset because it offers rich information (dataset size: 20k records) that allows us to classify the appropriate queue for each support ticket based on the given details. By analyzing the info given, we aim to predict which team or department (queue) should handle the ticket.     

Furthermore, data was enriched using `all_tickets.csv`: [Customer-Support-Ticket-Classification](https://github.com/Er-Devanshu/Customer-Support-Ticket-Classification/tree/main)
 
This classification task can help to automatize the assignment process in customer support systems, ensuring that each ticket is directed to the right team for a timely and effective response.

## Project Structure

```
.
├── data/                  # Dataset files
├── models/                # Trained models and model artifacts
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   ├── CNN.ipynb          # CNN model training and evaluation
│   ├── DistilBERT.ipynb   # DistilBERT model training and evaluation
│   └── NB_SVM.ipynb       # Naive Bayes and SVM models training and evaluation
├── src/                   # Source code
│   ├── __init__.py          
│   ├── data_exploration_utils.py   # EDA exploration
│   ├── evaluate_model.py           # Model evaluations
│   ├── models.py          # Models creation (except CNN)
│   ├── tf_init.py         # Tensorflow
│   └── utils.py           # Useful load functions
├── streamlit/             # Streamlit App
│   ├── app.py             # Streamlit App code with CNN model
│   └── README_APP.md      # Contains information about the App
├── README.md              # This file
└── requirements.txt       # Evinoment's requirements
```

## Key Features

- **Text Processing**: Tokenization, stemming, stopword removal, and TF-IDF vectorization.
- **Multiple Models**: Support for CNN, DistilBERT, Naive Bayes and SVM classifiers.
- **Hyperparameter Tuning**: Grid search for optimal model parameters.
- **Evaluation**: Comprehensive evaluation metrics including confusion matrices, classification reports, and visualizations.
- **Model Persistence**: Save and load trained models for reuse.
- **Exploratory Analysis**: Tools for understanding data distribution and text characteristics.

## Results
Each notebook provides an explanation for every decision made, along with a comprehensive analysis of the results achieved.
