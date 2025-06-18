from django.shortcuts import render
import joblib
import os
from .utils import clean_text

# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vectorizer=joblib.load(r'C:\Users\sudha\PycharmProjects\Sentiment-analysis\vectorizer.pkl')
model_sentiment=joblib.load(r'C:\Users\sudha\PycharmProjects\Sentiment-analysis\sentiment.pkl')

def predict_sentiment(request):
    prediction=None
    if request.method=='POST':
        review=request.POST.get('review')
        review_cleaned=clean_text(review)
        review_vectorized=vectorizer.transform([review_cleaned])
        result=model_sentiment.predict(review_vectorized)[0]
        prediction="Positive" if result==1 else "Negative"

    return render(request, 'analyzer/index.html', {'prediction':prediction})