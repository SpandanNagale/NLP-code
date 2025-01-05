import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup as beautifulsoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



df=pd.read_csv("all_kindle_review.csv")
df=df[['rating','reviewText']]

df['rating']=df['rating'].apply(lambda x:0 if x<3 else 1)
df['reviewText']=df['reviewText'].str.lower()
df['reviewText']=df['reviewText'].apply(lambda x:re.sub('[^a-z A-Z 0-9]+', " ",x))
df['reviewText']=df['reviewText'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))
df['reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
df['reviewText']=df['reviewText'].apply(lambda x: beautifulsoup(x,'lxml').get_text())
df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x.split()))


nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

df['reviewText'] = df['reviewText'].apply(lemmatize_text)


X_train, X_test, y_train, y_test = train_test_split(df["reviewText"],df['rating'],test_size=0.20)




BOW=CountVectorizer()
Tfidf=TfidfVectorizer()

x_train_BOW=BOW.fit_transform(X_train).toarray()
x_test_BOW=BOW.transform(X_test).toarray()
x_train_TFIDF=Tfidf.fit_transform(X_train).toarray()
x_test_TFIDF=Tfidf.transform(X_test).toarray()




model=GaussianNB()

model.fit(x_train_BOW,y_train)

y_pred=model.predict(x_test_BOW)
print(accuracy_score(y_test,y_pred))
model.fit(x_train_TFIDF,y_train)
y_pred=model.predict(x_test_TFIDF)
print(accuracy_score(y_test,y_pred))


model=MultinomialNB()

model.fit(x_train_BOW,y_train)

y_pred=model.predict(x_test_BOW)
print(accuracy_score(y_test,y_pred))
model.fit(x_train_TFIDF,y_train)
y_pred=model.predict(x_test_TFIDF)
print(accuracy_score(y_test,y_pred))