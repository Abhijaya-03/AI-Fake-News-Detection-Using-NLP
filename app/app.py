from flask import Flask, render_template, request ,session
import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.secret_key = "fake_news_secret_key"

# Load saved model and vectorizer
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


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    result_class = ""

    if request.method == "POST":

      news = request.form["news"]

      cleaned_news = clean_text(news)

      news_vector = vectorizer.transform([cleaned_news])

      result = model.predict(news_vector)
      
      print("Prediction value:", result[0])

      if result[0] == 0:
        prediction = " Fake News Detected"
        result_class = "result-fake"
      else:
        prediction = " Real News Detected"
        result_class = "result-real"
      if "history" not in session:
       session["history"] = []

      history = session["history"]

      history.append(prediction)

      session["history"] = history[-5:]
    return render_template(
    "index.html",
    prediction=prediction,
    result_class=result_class,
    history=session.get("history", [])
)

if __name__ == "__main__":
    app.run(debug=True)
