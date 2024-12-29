import pandas
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
paragraph="""I have three visions for India. In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Mughals, the Portuguese, the British, the French, the Dutch  all of them came and looted us, took over what was ours. Yet, we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture, their history, and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That is why I have a vision for India.

I believe that India got its first vision of freedom in 1857 when we started the War of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.

My second vision for Indias development. For fifty years, we have been a developing nation. It is time we see ourselves as a developed nation. We are among the top 5 nations in the world in terms of GDP. We have a growth rate of 10% plus. Our poverty levels are falling. Our achievements are being globally recognized today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant, and self-assured. Isnt this incorrect?

I have a third vision. India must stand up to the world. Because I believe that unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong, not only as a military power but also as an economic power. Both must go hand-in-hand.

Why is the media in India so negative? Why are we, in India, so embarrassed to recognize our own strengths, our achievements? We are such a great nation. We have so many amazing success stories, but we refuse to acknowledge them. Why?

I am echoing J.F. Kennedys words to his fellow Americans to relate to Indians today: "Ask what we can do for India and do what has to be done to make India what America and other western countries are today."

Lets work together to create a strong, developed, and proud nation. Let us dream big and transform those dreams into reality."""



from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stemmer=PorterStemmer()
lemmatizer=WordNetLemmatizer()

sentences=nltk.sent_tokenize(paragraph)

for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    #words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)

print(sentences)
