# titanic_eda.py

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import logging

# =============== Config ===============
DATA_PATH = 'titanic.csv'
OUTPUT_DIR = 'plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="darkgrid")
plt.rcParams['figure.dpi'] = 130

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============== Helpers ===============

def save_and_show(fig, filename):
    """Save figure to OUTPUT_DIR and show it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# =============== EDA Functions ===============

def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        exit(1)

def summarize(df):
    logging.info("Data Info:")
    print(df.info())
    logging.info("Missing Values:")
    print(df.isnull().sum())
    logging.info("Duplicate Rows:")
    print(df.duplicated().sum())
    logging.info("Descriptive Statistics:")
    print(df.describe(include='all').T)

def plot_missing_values(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, palette="viridis", ax=ax)
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    save_and_show(fig, 'missing_values.png')

def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap")
    save_and_show(fig, 'missing_values_heatmap.png')

def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    save_and_show(fig, 'correlation_heatmap.png')

def plot_distributions(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, color='teal', ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_and_show(fig, f'dist_{col}.png')

def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], color='coral', ax=ax)
        ax.set_title(f"Boxplot of {col}")
        save_and_show(fig, f'boxplot_{col}.png')

def plot_survival_by_category(df, cat_col):
    if 'Survived' not in df.columns:
        logging.warning("No 'Survived' column to analyze survival.")
        return
    fig, ax = plt.subplots()
    sns.countplot(x=cat_col, hue='Survived', data=df, palette='Set1', ax=ax)
    ax.set_title(f"Survival Count by {cat_col}")
    plt.xticks(rotation=0)
    save_and_show(fig, f'survival_by_{cat_col}.png')

def gender_survival_ttest(df):
    if not {'Sex', 'Survived'}.issubset(df.columns):
        logging.warning("Required columns for t-test missing.")
        return
    male_survived = df.loc[df['Sex'] == 'male', 'Survived'].dropna()
    female_survived = df.loc[df['Sex'] == 'female', 'Survived'].dropna()
    t_stat, p_val = ttest_ind(male_survived, female_survived)
    logging.info(f"T-test on Survival by Gender: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        logging.info("Statistically significant difference detected.")
    else:
        logging.info("No statistically significant difference detected.")

# =============== Main ===============

def main():
    df = load_data(DATA_PATH)

    # Convert relevant columns to categorical for better plots
    for col in ['Sex', 'Pclass', 'Embarked']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # 1. Summarize Data
    summarize(df)

    # 2. Missing Data Visualizations
    plot_missing_values(df)
    plot_missing_heatmap(df)

    # 3. Correlation Heatmap
    plot_correlation_heatmap(df)

    # 4. Numeric Feature Distributions & Outliers
    plot_distributions(df)
    plot_boxplots(df)

    # 5. Survival Analysis by Category
    for cat in ['Sex', 'Pclass', 'Embarked']:
        if cat in df.columns:
            plot_survival_by_category(df, cat)

    # 6. Statistical Test: Survival difference by Gender
    gender_survival_ttest(df)

    logging.info("EDA complete! Plots saved in 'plots/' folder.")

if __name__ == "__main__":
    main()
