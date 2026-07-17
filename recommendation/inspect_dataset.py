import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Abrar\Downloads\products_dataset.csv")

# First 5 rows
print(df.head())

print("\n------------------------")

# Dataset information
print(df.info())

print("\n------------------------")

# Missing values
print(df.isnull().sum())

print("\n------------------------")

# Number of duplicate rows
print("Duplicate rows:", df.duplicated().sum())

print("\n------------------------")

# Column names
print(df.columns)