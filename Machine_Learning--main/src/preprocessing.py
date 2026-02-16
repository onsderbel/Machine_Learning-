# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Paths
RAW_DATA_PATH = "data/raw/RetailCustomers.csv"
OUTPUT_PATH = "data/train_test/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features only
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save processed data
X_train.to_csv(OUTPUT_PATH + "X_train.csv", index=False)
X_test.to_csv(OUTPUT_PATH + "X_test.csv", index=False)
y_train.to_csv(OUTPUT_PATH + "y_train.csv", index=False)
y_test.to_csv(OUTPUT_PATH + "y_test.csv", index=False)

print("✅ Preprocessing completed. Train/Test files saved.")
