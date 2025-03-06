import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    current_directory = os.getcwd()
    filename = "data.csv"
    file_path = os.path.join(current_directory, filename)

    if not os.path.exists(file_path):
        print(
            f"Error: Dataset not found at {file_path}. Please upload "
            f"'{filename}' to the current directory."
        )
        exit(1)

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(
            f"Error reading file: {e}. Please ensure the file is correctly formatted."
        )
        exit(1)


def preprocessing(df):
    df.columns = [
        col.strip().replace(" ", "_").lower() for col in df.columns
    ]

    if df.isnull().sum().any():
        print("\nMissing values detected:")
        print(df.isnull().sum())
        print("Dropping rows with missing values.")
        df = df.dropna()

    print("\nDataset Info:\n", df.info())
    print("\nFirst Few Rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())

    return df


def plot_relational_plot(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numerical_cols) < 2:
        print("Error: Not enough numerical columns for a scatter plot.")
        return

    x_col, y_col = numerical_cols[0], numerical_cols[1]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.savefig("relational_plot.png")
    plt.show()


def plot_categorical_plot(df):
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if not categorical_cols:
        print("Error: No categorical columns found for a bar chart.")
        return

    cat_col = categorical_cols[0]

    plt.figure(figsize=(8, 5))
    df[cat_col].value_counts().plot(kind="bar", color="skyblue")
    plt.xlabel(cat_col)
    plt.ylabel("Count")
    plt.title(f"Bar Chart of {cat_col}")
    plt.savefig("categorical_plot.png")
    plt.show()


def plot_statistical_plot(df):
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        print("Error: No numerical columns found for correlation heatmap.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[numerical_columns].corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    plt.savefig("statistical_plot.png")
    plt.show()


def statistical_analysis(df, col):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    return mean, stddev, skew, excess_kurtosis


def writing(moments, col):
    mean, stddev, skew, excess_kurtosis = moments

    print(f"\nFor the attribute {col}:")
    print(
        f"Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, "
        f"Skewness = {skew:.2f}, Excess Kurtosis = {excess_kurtosis:.2f}."
    )

    skewness_desc = (
        "not skewed"
        if abs(skew) < 0.5
        else ("right-skewed" if skew > 0 else "left-skewed")
    )
    kurtosis_desc = (
        "mesokurtic"
        if -2 < excess_kurtosis < 2
        else ("leptokurtic" if excess_kurtosis > 2 else "platykurtic")
    )

    print(f"The data is {skewness_desc} and {kurtosis_desc}.\n")


def main():
    df = load_data()
    df = preprocessing(df)

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        print("Error: No numerical columns found for analysis.")
        exit(1)

    col = numerical_columns[0]

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == "__main__":
    main()
