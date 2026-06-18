import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load datasets
fake = pd.read_csv("../datasets/Fake.csv")
real = pd.read_csv("../datasets/real.csv")

# Labels
fake["label"] = 0
real["label"] = 1

# Merge
data = pd.concat([fake, real], ignore_index=True)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["content"].apply(clean_text)

print("Dataset Shape:")
print(data.shape)

print("\nOriginal Text:")
print(data["content"].iloc[0][:300])

print("\nCleaned Text:")
print(data["clean_text"].iloc[0][:300])