import pandas as pd

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