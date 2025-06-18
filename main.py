import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from matplotlib.style.core import library
from nltk.tokenize import word_tokenize
from numpy.ma.core import negative
from scipy.optimize import anderson

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib


def load_data(data_dir, max_files):
    data=[]

    for label_type in ['pos','neg']:
        label=1 if label_type=='pos' else 0 #label 1 for happy and 0 for sad
        folder=os.path.join(data_dir,label_type)

        allfiles=[f for f in os.listdir(folder) if f.endswith('.txt')]
        files_to_read=allfiles[:max_files] #This max_files_per_class is slicing start from 0 to max

        for filename in files_to_read:
            file_path = os.path.join(folder, filename)
            with open(file_path, encoding="utf8") as f:
                review = f.read()
                data.append((review, label))

    return pd.DataFrame(data, columns=['review', 'sentiment'])

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

train_data='aclImdb/train'
test_data='aclImdb/test'

train_dFrame=load_data(train_data, max_files=12500)
test_dFrame=load_data(test_data, max_files=12500)

# print("Train data,\n\n")
# print(train_dFrame.tail())
# print("Test data,\n\n")
# print(test_dFrame.tail())

train_dFrame['clean_review']=train_dFrame['review'].apply(clean_text)
test_dFrame['clean_review']=test_dFrame['review'].apply(clean_text)

# print("Train data,\n\n")
# print(train_dFrame.head())
# print("Test data,\n\n")
# print(test_dFrame.tail())

#This is BoW and ngram vectorization gave around 0.799 accuracy so we will try better model
# cv=CountVectorizer(max_features=20000, ngram_range=(1,2))

#The better model i.e Tf idf vectorization does some math calculation and gives value
cv=TfidfVectorizer(max_features=20000, ngram_range=(1,2))
# Starting let's try BoW
train_bow=cv.fit_transform(train_dFrame['clean_review'])
test_bow=cv.transform(test_dFrame['clean_review'])

# print(train_bow[0].toarray())
# print(test_bow[0].toarray())

X_train=train_bow
y_train=train_dFrame['sentiment']
X_test=test_bow
y_test=test_dFrame['sentiment']

# Used LogisticRegression which is USED FOR binary classification doesn't give that nice result so,
# model=LogisticRegression()

# Using svc no better difference tho
# model = LinearSVC()

model = MultinomialNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(f"Report: \n {classification_report(y_test, predictions)}")


# DATA VISUALISATION
cm=confusion_matrix(y_test, predictions)

#plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CONFUSION MATRIX")
plt.tight_layout()
plt.show()

joblib.dump(cv, 'vectorizer.pkl')
# joblib library is designed to save machine learning models and
# and large objects efficiently, this way,
# You don't have to retrain your model and can load them in the website backend
joblib.dump(model, 'sentiment.pkl')
# here .pkl saves snapshots of my trained compenents like;
# baad mein inko joblib.load karenge