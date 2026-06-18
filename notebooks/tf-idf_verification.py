import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

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

# Clean text
data["clean_text"] = data["content"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
# X contains the TF-IDF representation of the news articles
X = tfidf.fit_transform(data["clean_text"])
# y contains the target labels (0 for fake, 1 for real)
y = data["label"]

print("Feature Matrix Shape:")
print(X.shape)

print("\nLabels Shape:")
print(y.shape)
print("\nSample Feature Names:")
print(tfidf.get_feature_names_out()[:20])