import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))

def check_class_imbalance(df, column_name):
    """
    Calculates and returns the frequency and percentage (rounded to 2 decimals)
    of each class in the specified column of the dataset.
 
    Inputs:
        df (DataFrame): The dataset containing the column.
        column_name (str): The name of the column to analyze.
 
    Returns:
        DataFrame: A dataframe with the counts and percentages of each class.
    """

    # Calculate the frequency of each class
    class_counts = df[column_name].value_counts()

    # Calculate the percentage of each class
    class_percentage = (df[column_name].value_counts(normalize=True) * 100).round(2)

    # Create a DataFrame to display counts and percentages
    df_final= pd.DataFrame({'Counts': class_counts, 'Percentage': class_percentage})

    return df_final

def preprocess_text(text, language='en'):
    """
    Preprocesses a given text by applying several text cleaning steps, such as:
    - Lowercasing the text
    - Expanding contractions (English only)
    - Removing punctuation and numbers
    - Tokenizing the text
    - Removing stopwords (based on the specified language)
    - Lemmatizing the tokens
    - Removing very short tokens (e.g., noise or irrelevant words)

    Inputs:
        text (str): The input text to be preprocessed.
        language (str, optional) [default = 'en']: The language of the text. Supports 'en' for English and 'de' for German.

    Returns:
        str: The cleaned and preprocessed text as a single string.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions (English only)
    if language == 'en':
        text = contractions.fix(text)

    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords (according to language)
    if language == 'en':
        tokens = [t for t in tokens if t not in stop_words_en]
    elif language == 'de':
        tokens = [t for t in tokens if t not in stop_words_de]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Remove very short tokens (noise)
    tokens = [t for t in tokens if len(t) > 2]

    return ' '.join(tokens)

def generate_wordcloud(df, text_column='clean_text', stopwords=None, width=800, height=400, colormap='coolwarm'):
    """
    Generates and displays a word cloud from the given dataframe.

    Inputs: 
        df: The DataFrame containing the text data.
        text_column (str) [default = 'clean_text']: The column containing the text for the word cloud.
        stopwords (set, optional): A set of stopwords to exclude from the word cloud.
        width (int) [default = 800]: The width of the word cloud image.
        height (int) [default = 400]: The height of the word cloud image.
        colormap (str) [default = 'coolwarm']: The color map for the word cloud.

    Returns:
        None
    """
    # Combine all text into one large string
    text_data = ' '.join(df[text_column].dropna())

    # Tokenize and filter
    tokens = text_data.split()
    
    if stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]

    frequencies = Counter(tokens)
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap=colormap
    ).generate_from_frequencies(frequencies)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off') 
    plt.title("Most Frequent Words in the Dataset", fontsize=16)
    plt.show()
