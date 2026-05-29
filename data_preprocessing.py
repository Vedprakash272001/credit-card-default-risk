import pandas as pd
import numpy as np

print("--- Starting Task 2: Data Preprocessing ---")

df = pd.read_csv('Credit_Card.csv', sep=';')

# Handling Missing Values 

df['LIMIT_BAL'] = df['LIMIT_BAL'].fillna(df['LIMIT_BAL'].median())
df['AGE'] = df['AGE'].fillna(df['AGE'].median())

df['SEX'] = df['SEX'].fillna(df['SEX'].mode()[0])
df['EDUCATION'] = df['EDUCATION'].fillna(df['EDUCATION'].mode()[0])
df['MARRIAGE'] = df['MARRIAGE'].fillna(df['MARRIAGE'].mode()[0])
df['PAY_AMT1'] = df['PAY_AMT1'].fillna(0)  
df['PAY_AMT2'] = df['PAY_AMT2'].fillna(0)

df['LIMIT_BAL_LOG'] = df['LIMIT_BAL_LOG'].astype(str).str.replace('.', '', n=2)
df['LIMIT_BAL_LOG'] = pd.to_numeric(df['LIMIT_BAL_LOG'], errors='coerce')
df['LIMIT_BAL_LOG'] = df['LIMIT_BAL_LOG'].fillna(df['LIMIT_BAL_LOG'].median())

df = pd.get_dummies(df, columns=['CITY'], drop_first=True)

print("\nRemaining Missing Values Check:")
print(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE']].isnull().sum())
print(f"\nCleaned Dataset Shape: {df.shape}")

df.to_csv('Cleaned_Credit_Card.csv', index=False)
print("\nSuccess! Cleaned data saved as 'Cleaned_Credit_Card.csv'")