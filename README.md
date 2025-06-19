# Sentiment Analysis - NLP Web App

This is a web application built using traditional NLP techniques and Django (Python backend), where users can input a sentence and get real-time sentiment analysis results (Positive or Negative). The app is integrated with a REST API and is powered by machine learning models trained on preprocessed text data using Bag of Words and TF-IDF techniques.

## Features

- **Sentiment Prediction**: Users can enter any sentence, and the app will classify it as either Positive or Negative.
- **Trained Models**: Supports multiple ML algorithms like Logistic Regression, Naive Bayes, and SVC (Logistic Regression with BoW was used in production).
- **Joblib Serialization**: The trained model and vectorizer are saved using `joblib` for fast loading during inference.
- **REST API Support**: Provides a `/api/analyze/` endpoint to analyze sentiment via HTTP POST requests.
- **Responsive UI**: Simple and clean Bootstrap-based frontend interface.
- **Error Handling**: Basic error feedback for invalid inputs.

## Tech Stack

- **Frontend**: HTML, Bootstrap
- **Backend**: Django, Django REST Framework
- **Machine Learning**: scikit-learn, NLTK
- **Serialization**: joblib
- **Deployment**: Render (currently inactive due to free-tier limitations)

## API Endpoint

- **POST** `/api/analyze/`
- **Request Body**:

## Project Structure
 sentiment-analysis/
 
├── api/                  # Django app for API handling

├── model/                # Serialized ML model & vectorizer

│   ├── sentiment_model.pkl

│   └── vectorizer.pkl

├── static/               # Static assets (CSS, images)

├── templates/            # HTML frontend templates

├── sentiment/            # Main Django project

├── manage.py

└── requirements.txt

Made by Sudhanshu Anand

GitHub: @Theallkeeeymist

Email: sudhanshuanand4529@gmail.com
