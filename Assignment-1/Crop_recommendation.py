import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('crop_recommendation.csv')

print("Data Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumerical Columns:", list(numerical_cols))
print("Categorical Columns:", list(categorical_cols))

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numerical Variables)')
plt.show()

if len(categorical_cols) > 0:
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(categorical_cols):
        plt.subplot(1, len(categorical_cols), i+1)
        value_counts = df[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

sns.pairplot(df[numerical_cols], diag_kind='kde')
plt.show()

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    if col != 'label': 
        plt.subplot(3, 3, i+1)
        sns.boxplot(data=df, x='label', y=col)
        plt.xticks(rotation=45)
        plt.title(f'{col} by Crop Type')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='label')
plt.xticks(rotation=45)
plt.title('Distribution of Crop Types')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='temperature', y='humidity', hue='label')
plt.title('Temperature vs Humidity for Different Crops')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, nutrient in enumerate(['N', 'P', 'K']):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(data=df, x='rainfall', y=nutrient)
    plt.title(f'Rainfall vs {nutrient}')
plt.tight_layout()
plt.show()