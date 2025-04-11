import pandas as pd
import re
import string
from langdetect import detect
from deep_translator import GoogleTranslator
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

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


def translate_all_to_english(df, text_column='text'):
    """
    Traduce el contenido de la columna especificada de cada registro del DataFrame a inglés.
    Se utiliza la detección automática del idioma de origen, de modo que aun si un registro está en inglés,
    se traducirá (lo que en la mayoría de los casos devolverá el mismo texto).
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los registros.
        text_column (str): Nombre de la columna que contiene el texto a traducir.
    
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'text_en' que contiene el texto traducido a inglés.
    """
    
    # Creación del traductor con detección automática del idioma de origen.
    translator = GoogleTranslator(source='auto', target='en')
    
    def translate_row(text):
        try:
            # Se traduce el texto a inglés
            translated_text = translator.translate(text)
            return translated_text
        except Exception as e:
            print(f"Error al traducir el texto: {e}")
            # Si ocurre algún error, se retorna el texto original
            return text

    # Aplicar la función de traducción a la columna especificada
    df['text_en'] = df[text_column].apply(translate_row)
    return df

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

def preprocess_text(text):
    """
    Preprocesses English text by applying several cleaning steps:
    - Lowercasing the text
    - Expanding contractions
    - Removing punctuation and numbers
    - Tokenizing the text
    - Removing English stopwords
    - Lemmatizing the tokens
    - Removing very short tokens (e.g., noise or irrelevant words)

    Args:
        text (str): The input English text to be preprocessed.

    Returns:
        str: The cleaned and preprocessed text as a single string.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)

    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove English stopwords
    tokens = [t for t in tokens if t not in stop_words_en]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Remove very short tokens (e.g., <=2 characters)
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
