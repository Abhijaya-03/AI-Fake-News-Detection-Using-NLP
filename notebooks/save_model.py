import pandas as pd
import re
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# Load the datasets
# ==========================================

fake = pd.read_csv("../datasets/Fake.csv")
real = pd.read_csv("../datasets/real.csv")


# ==========================================
# Create labels
# Fake = 0
# Real = 1
# ==========================================

fake["label"] = 0
real["label"] = 1


# ==========================================
# Merge datasets
# ==========================================

data = pd.concat([fake, real], ignore_index=True)


# ==========================================
# Combine title and article text
# ==========================================

data["content"] = data["title"] + " " + data["text"]


# ==========================================
# Text preprocessing setup
# ==========================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# ==========================================
# Apply preprocessing
# ==========================================

data["clean_text"] = data["content"].apply(clean_text)


# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])

y = data["label"]


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Train Passive Aggressive Classifier
# ==========================================

pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(X_train, y_train)


# ==========================================
# Model Evaluation
# ==========================================

y_pred = pac.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:")
print(round(accuracy * 100, 2), "%")


# ==========================================
# Save Model and Vectorizer
# ==========================================

joblib.dump(pac, "../model/fake_news_model.pkl")

joblib.dump(vectorizer, "../model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")