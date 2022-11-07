import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("fake_or_real_news.csv")

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)

# Splitting the data into training and testing.
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
#Training the model.
model.fit(xtrain, ytrain)


news_headline = input("Enter the news headline: ")

data = cv.transform([news_headline]).toarray()

print(model.predict(data))