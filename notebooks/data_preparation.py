import pandas as pd

# Load datasets
fake = pd.read_csv("../datasets/Fake.csv")
real = pd.read_csv("../datasets/real.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

print("Combined Dataset Shape:")
print(data.shape)

print("\nLabel Distribution:")
print(data["label"].value_counts())

print("\nFirst 5 Rows:")
print(data.head())