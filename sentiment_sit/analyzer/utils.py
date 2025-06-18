import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(review):
    review=review.lower() #'Good' ban gaya 'good'
    review=BeautifulSoup(review,"html.parser").get_text() #HTML tags hataya
    review=re.sub(r'[^a-zA-Z]', ' ', review) #replace everything except alphabets into space
    words=review.split() #I Love AI ban gaya ['I', 'Love', 'AI']
    stop_words=set(stopwords.words("english")) #is am the in and etc
    words=[w for w in words if w not in stop_words]
    lemmatizer=WordNetLemmatizer()
    words=[lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)