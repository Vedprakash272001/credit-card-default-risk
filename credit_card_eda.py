import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Credit_Card.csv', sep=';')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())
print("\n--- Descriptive Statistics Summary ---")
print(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())
print("\n--- Default Class Distribution ---")
print(df['default.payment.next.month'].value_counts())
default_pct = df['default.payment.next.month'].mean() * 100
print(f"Percentage of Defaults: {default_pct:.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(x='default.payment.next.month', data=df, palette='Set2')
plt.title('Distribution of Credit Card Defaults')
plt.xlabel('Default Status (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('default_distribution.png')
print("\nSuccess! Chart saved as 'default_distribution.png'")