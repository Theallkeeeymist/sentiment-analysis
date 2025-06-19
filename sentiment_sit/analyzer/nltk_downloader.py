import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') #to use wordnetlemmetizer
nltk.download('omw-1.4')

# Start Command
# cd sentiment_sit && pip install -r .../requirements.txt && python analyzer/nltk_downloader.py && python manage.py collectstatic --noinput && gunicorn sentiment_sit.wsgi:application