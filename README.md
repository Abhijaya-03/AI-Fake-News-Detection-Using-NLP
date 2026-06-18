# AI-Based Fake News Detection System

## Project Overview

The AI-Based Fake News Detection System is a machine learning web application developed to classify news articles as Real or Fake using Natural Language Processing (NLP) techniques and machine learning algorithms.

The system preprocesses textual news content, converts it into numerical features using TF-IDF Vectorization, and predicts whether the news is genuine or misleading using a Passive Aggressive Classifier. The trained model is deployed through a Flask-based web interface that allows users to analyze news articles in real time.

---
## Objectives

* Detect fake and misleading news articles.
* Apply Natural Language Processing techniques for text preprocessing.
* Train a machine learning model for classification.
* Deploy the trained model using Flask.
* Provide a simple and user-friendly interface for news verification.

---
## Features

* Real News Detection
* Fake News Detection
* NLP-based Text Preprocessing
* TF-IDF Vectorization
* Passive Aggressive Classifier
* Word Counter
* Clear Text Functionality
* Prediction History
* Responsive User Interface

---
## Technologies Used

* Python
* Flask
* Scikit-learn
* NLTK
* TF-IDF Vectorization
* HTML
* CSS

---
## Dataset

The project uses two datasets:

* Real.csv
* Fake.csv
These datasets contains real and fake news articles used for training and testing the machine learning model.

---
## Project Workflow

1. Collect Real and Fake News datasets.
2. Clean and preprocess text data.
3. Remove stopwords and perform lemmatization.
4. Convert text into numerical features using TF-IDF.
5. Train the Passive Aggressive Classifier.
6. Save the trained model and vectorizer.
7. Deploy the model using Flask.
8. Predict whether the news article is Real or Fake.

---
## Project Structure

├──app/

 └──templates/

│    └── index.html

├── app.py

├── requirements.txt

├── datasets/

│ ├── Fake.csv

│ └── Real.csv

├── model/

│ ├── fake_news_model.pkl

│ └── tfidf_vectorizer.pkl

├── notebooks/

│ ├── dataset_exploration.py

│ ├── dataset_cleaning.py

│ ├── text_preprocessing.py

│ ├── data_preparation.py

│ ├── tf-idf_verification.py

│ ├── pac_training.py

│ ├── save_model.py

│ └── predict_news.py

└── report/

---
## Installation

1. Clone the repository.

2. Install required libraries:

pip install -r requirements.txt

3. Run the Flask application:

python app.py

4. Open the application in your browser:

http://127.0.0.1:5000

## Screenshots
# Home Page
![Home Page](screenshots/home_pg.png)

# Real News Detection
![Real News Detection](screenshots/real_news.png)

# Fake News Detection
![Fake News Detection](screenshots/fake_news.png)

---
## Results

The system successfully classifies news articles as Real or Fake using machine learning and NLP techniques. The Flask-based interface provides real-time predictions and maintains a prediction history for user convenience.

---
## Future Enhancements

* BERT-based classification model
* Confidence score visualization
* News source credibility checking
* Cloud deployment
* User authentication

---
## Developer
Abhijaya

AI-Based Fake News Detection System

Machine Learning | NLP | Flask
