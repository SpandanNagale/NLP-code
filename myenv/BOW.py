import pandas as pd
messages=pd.read_csv('SMSSpamCollection' , sep='\t' , names=['label','message'])
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]', " ",messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)

cv=CountVectorizer(max_features=100,ngram_range=(1,2))
x=cv.fit_transform(corpus).toarray()