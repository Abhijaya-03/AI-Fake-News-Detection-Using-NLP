import pandas as pd
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


# Load the datasets
fake = pd.read_csv("../datasets/Fake.csv")
real = pd.read_csv("../datasets/real.csv")

# -----------------------------
# Create labels
# Fake = 0
# Real = 1
# -----------------------------
fake["label"] = 0
real["label"] = 1

# Merge datasets
data = pd.concat([fake, real], ignore_index=True)


# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# NLP preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# Clean all articles
data["clean_text"] = data["content"].apply(clean_text)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])

y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(X_train, y_train)

# Predictions
y_pred = pac.predict(X_test)

# Accuracy-
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))