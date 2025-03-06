import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss


def plot_relational_plot(df):
    """
    Generates a scatter plot to visualize relationships between two numerical variables.
    Saves the figure as 'relational_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[df.columns[0]], y=df[df.columns[1]], ax=ax)  # Adjust column names if needed
    ax.set_title(f"Scatter Plot: {df.columns[0]} vs {df.columns[1]}")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Generates a bar chart for a categorical column.
    Saves the figure as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df[df.columns[2]].value_counts().plot(kind='bar', color='skyblue', ax=ax)  # Adjust column name if needed
    ax.set_title(f"Bar Chart of {df.columns[2]}")
    plt.xlabel(df.columns[2])
    plt.ylabel("Count")
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    Generates a correlation heatmap to show relationships between numerical variables.
    Saves the figure as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col: str):
    """
    Computes and returns key statistical moments for a given numerical column.

    Parameters:
    df (pd.DataFrame): The dataset
    col (str): The column name for statistical analysis

    Returns:
    tuple: Mean, Standard Deviation, Skewness, Excess Kurtosis
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataset by handling missing values and standardizing column names.

    Parameters:
    df (pd.DataFrame): The raw dataset

    Returns:
    pd.DataFrame: The cleaned dataset
    """
    df = df.dropna()  # Remove missing values
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]  # Standardize column names

    # Display basic dataset information
    print("Dataset Information:\n", df.info())
    print("\nFirst Few Rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr())

    return df


def writing(moments, col):
    """
    Prints the computed statistical moments in a formatted manner.

    Parameters:
    moments (tuple): Mean, Standard Deviation, Skewness, and Kurtosis
    col (str): The column name being analyzed
    """
    mean, stddev, skew, excess_kurtosis = moments

    print(f'For the attribute {col}:')
    print(f'Mean = {mean:.2f}, '
          f'Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, and '
          f'Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Interpretation of skewness and kurtosis
    skewness_description = "not skewed" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
    kurtosis_description = "mesokurtic" if -2 < excess_kurtosis < 2 else ("leptokurtic" if excess_kurtosis > 2 else "platykurtic")

    print(f'The data is {skewness_description} and {kurtosis_description}.\n')


def main():
    """
    Main function that loads the dataset, processes it, generates plots,
    performs statistical analysis, and prints the findings.
    """
    df = pd.read_csv('Data.csv')  # Ensure your dataset is named correctly
    df = preprocessing(df)

    col = df.columns[0]  # Change this to the column you want to analyze

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()

    
