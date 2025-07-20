# model/train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_imdb_data(directory):
    data = []
    for label in ["pos", "neg"]:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding="utf-8") as f:
                text = f.read()
                data.append((text, "Positive" if label == "pos" else "Negative"))
    return pd.DataFrame(data, columns=["text", "sentiment"])

# Load training data
train_df = load_imdb_data("data/aclImdb/train")

# Shuffle and split
X_train, X_test, y_train, y_test = train_test_split(
    train_df["text"], train_df["sentiment"], test_size=0.2, random_state=42
)

# Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=5)),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model/model.pkl")

print("✅ New IMDb model trained and saved.")
