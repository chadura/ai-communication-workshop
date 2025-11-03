import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


# Load data
data = pd.read_csv('data/sms_spam_collection.csv', sep='\t', names=['label', 'message'])
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)


# Vectorize and train
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Evaluate
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, y_pred))


# Save artifacts
joblib.dump((vectorizer, model), 'model.joblib')
print('Model saved to model.joblib')