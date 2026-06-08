import pandas as pd
import numpy as np

print("--- Upgrading Task 2: Advanced Matching Preprocessing ---")

# 1. Load raw data
df = pd.read_csv('Credit_Card.csv', sep=';')
print(f"Initial Raw Shape: {df.shape}")

# 2. DROP DUPLICATES (Removes messy repeat observations)
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# 3. FIRST: Clean the malformed string text formatting anomalies globally
format_messy_cols = ['LIMIT_BAL_LOG', 'risk_leak']
for col in format_messy_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. NOW: Safe Missing Values Imputation (Using Median/Mode math)
num_cols = ['LIMIT_BAL', 'AGE', 'PAY_AMT1', 'PAY_AMT2', 'LIMIT_BAL_LOG', 'risk_leak']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. OUTLIER CAPPING (Percentile-based clipping matching your friend's logic)
cap_cols = ['LIMIT_BAL', 'LIMIT_BAL_LOG', 'risk_leak']
for col in cap_cols:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower_limit, upper_limit)

# 6. One-Hot Encoding for CITY text attributes
df = pd.get_dummies(df, columns=['CITY'], drop_first=True)

# 7. Save out the optimized dataset
df.to_csv('Cleaned_Credit_Card.csv', index=False)
print(f"Success! Cleaned data saved. Final Shape: {df.shape}")