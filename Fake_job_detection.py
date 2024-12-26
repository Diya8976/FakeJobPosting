import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import joblib

# Data preprocessing functions
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text

def preprocess_data(data):
    data['description'] = data['description'].fillna("").apply(preprocess_text)
    data['requirements'] = data['requirements'].fillna("").apply(preprocess_text)
    data['combined_text'] = data['description'] + " " + data['requirements']
    return data

# Feature extraction
def extract_features(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['combined_text'])
    return X, vectorizer

# Model training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, vectorizer, X_test, y_test

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Main script
if __name__ == "__main__":
    file_path = 'C:/Users/diya/Desktop/jupyter projects/fake_job_posts/fake_job_postings.csv/fake_job_postings.csv'


    data = pd.read_csv(file_path)
    data = preprocess_data(data)
    X, vectorizer = extract_features(data)
    y = data['fraudulent']  # Assuming the target column is 'fraudulent'

    model, vectorizer, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

    # Save the model and vectorizer
    joblib.dump(model, 'fake_job_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model and vectorizer saved successfully.")