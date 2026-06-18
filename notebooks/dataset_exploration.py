import pandas as pd

fake = pd.read_csv("../datasets/Fake.csv")
true = pd.read_csv("../datasets/real.csv")

print("Fake News Dataset Shape:")
print(fake.shape)

print("\nTrue News Dataset Shape:")
print(true.shape)

print("\nColumns:")
print(fake.columns)

print("\nFirst 5 Fake News Records:")
print(fake.head())

print("\nFirst 5 True News Records:")
print(true.head())