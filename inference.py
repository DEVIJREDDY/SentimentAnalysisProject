import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
texts = [
    "I love this product",
    "This is terrible"
]

labels = [
    "positive",
    "negative"
]

# Vectorize
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()

model.fit(X, labels)

# Test sentence
test = ["Amazing experience"]

test_vector = vectorizer.transform(test)

prediction = model.predict(test_vector)

print("Prediction:", prediction[0])