import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import nltk

nltk.download('stopwords', quiet=True)

# Load the dataset
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# BUG FIX #5: guard against unmapped labels / missing messages instead of
# silently feeding NaNs into training (defensive, even though current
# spam.csv happens to be clean).
data = data.dropna(subset=['label', 'message'])
data['label'] = data['label'].astype(int)

# BUG FIX #2: load stopwords ONCE as a set, not inside the per-word loop.
# stopwords.words('english') rebuilds a ~179-word list from disk on every
# call; doing that for every single word in every message made both
# training and live classification far slower than necessary.
STOP_WORDS = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    """Clean and stem a message for the bag-of-words model.

    BUG FIX #3: the old regex `[^a-zA-Z]` stripped every digit and symbol,
    throwing away strong spam signals like prices ("$1000"), percentages
    ("100% free"), and phone numbers. We now keep digits and the
    currency/percent symbols that commonly appear in spam, only dropping
    punctuation that adds noise.
    """
    text = re.sub(r'[^a-zA-Z0-9$%]', ' ', text)
    words = text.lower().split()
    stemmed = [ps.stem(w) for w in words if w not in STOP_WORDS]
    return ' '.join(stemmed)


corpus = [preprocess_text(msg) for msg in data['message']]

# Vectorization
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()
Y = data['label']

# BUG FIX #6: stratify the split so the train/test sets keep the same
# ~87/13 ham/spam ratio as the full dataset, instead of leaving it to chance.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# BUG FIX #1 + #7: address class imbalance with class_weight='balanced' so
# the model isn't biased toward predicting "ham", and set random_state so
# the trained model is reproducible across runs.
model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
)
model.fit(X_train, Y_train)

# Quick sanity check on held-out data
pred = model.predict(X_test)
print(classification_report(Y_test, pred, target_names=['ham', 'spam']))
print(confusion_matrix(Y_test, pred))

# Save the model and vectorizer
pickle.dump(model, open("spam_detector.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
