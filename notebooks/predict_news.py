import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved files
model = joblib.load("../model/fake_news_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")

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


# User input
news = input("Enter a news article or headline:\n")

# Clean text
cleaned_news = clean_text(news)

# Convert using saved TF-IDF vectorizer
news_vector = vectorizer.transform([cleaned_news])

# Prediction
prediction = model.predict(news_vector)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")
print(prediction)