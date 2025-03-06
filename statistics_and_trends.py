"""
Statistics and Trends Assignment

This script processes a dataset, performs statistical analysis, 
and generates relational, categorical, and statistical plots.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


def load_data():
    """
    Loads the dataset from 'Data.csv'. Checks if the file exists before loading.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    file_path = "Data.csv"  # Ensure this is the correct filename

    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}. Please upload 'Data.csv'.")
        exit(1)  # Stops execution if file is missing

    df = pd.read_csv(file_path)
    return df


def preprocessing(df):
    """
    Preprocesses the dataset by handling missing values and renaming columns.

    Parameters:
    df (pd.DataFrame): The raw dataset.

    Returns:
    pd.DataFrame: The cleaned dataset.
    """
    df = df.dropna()  # Remove missing values
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]  # Standardize column names

    # Display basic dataset information
    print("\nDataset Info:\n", df.info())
    print("\nFirst Few Rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())

    return df


def plot_relational_plot(df):
    """
    Generates a scatter plot for two numerical variables.
    Saves as 'relational_plot.png'.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[df.columns[0]], y=df[df.columns[1]])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(f"Scatter Plot: {df.columns[0]} vs {df.columns[1]}")
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Generates a bar chart for a categorical column.
    Saves as 'categorical_plot.png'.
    """
    plt.figure(figsize=(8, 5))
    df[df.columns[2]].value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel(df.columns[2])
    plt.ylabel("Count")
    plt.title(f"Bar Chart of {df.columns[2]}")
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    Generates a correlation heatmap for numerical variables.
    Saves as 'statistical_plot.png'.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col):
    """
    Computes key statistical moments for a numerical column.

    Parameters:
    df (pd.DataFrame): The dataset.
    col (str): The column name for analysis.

    Returns:
    tuple: Mean, Standard Deviation, Skewness, Excess Kurtosis.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    return mean, stddev, skew, excess_kurtosis


def writing(moments, col):
    """
    Prints statistical analysis results in a formatted manner.

    Parameters:
    moments (tuple): Mean, Std Dev, Skewness, Kurtosis.
    col (str): The column being analyzed.
    """
    mean, stddev, skew, excess_kurtosis = moments

    print(f'\nFor the attribute {col}:')
    print(f'Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Interpretation
    skewness_desc = "not skewed" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
    kurtosis_desc = "mesokurtic" if -2 < excess_kurtosis < 2 else ("leptokurtic" if excess_kurtosis > 2 else "platykurtic")

    print(f'The data is {skewness_desc} and {kurtosis_desc}.\n')


def main():
    """
    Main function to execute data processing, visualization, and analysis.
    """
    df = load_data()
    df = preprocessing(df)

    # Choose a numerical column for statistical analysis
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numerical_columns:
        print("Error: No numerical columns found for analysis.")
        exit(1)

    col = numerical_columns[0]  # Select the first numerical column

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
    
    
 
