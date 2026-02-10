# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(
    r"C:\Users\Komal Dixit\OneDrive\Desktop\ml\IMDB Dataset.csv"
)

# Show first 5 rows
print(df.head())

# -----------------------------
# Data Preprocessing
# -----------------------------
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Features and labels
x = df["review"]
y = df["sentiment"]

# -----------------------------
# Train-Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# -----------------------------
# Train Naive Bayes Model
# -----------------------------
nb = MultinomialNB()
nb.fit(x_train_tfidf, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = nb.predict(x_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review):
    review_tfidf = tfidf.transform([review])
    prediction = nb.predict(review_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

# -----------------------------
# Test Prediction
# -----------------------------
test_review = "the movie was boring and a complete waste of time"
print("\nReview:", test_review)
print("Predicted Sentiment:", predict_sentiment(test_review))
