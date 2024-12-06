import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import nltk

nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
ps = PorterStemmer()
corpus = []

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

# Vectorization
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()
Y = data['label']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Save the model and vectorizer
pickle.dump(model, open("spam_detector.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
