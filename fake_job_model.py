import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Preprocess the dataset
def preprocess_data(df):
    df = df.drop(columns=['salary_range', 'department'], errors='ignore')
    text_fields = ['description', 'company_profile', 'requirements', 'benefits']
    for field in text_fields:
        df[field] = df[field].fillna('').str.lower().str.replace(r'[^\w\s]', '', regex=True)
    print("Data preprocessing completed.")
    return df

# Build the ML pipeline
def build_pipeline():
    text_transformer = TfidfVectorizer(max_features=1000)
    
    pipeline = Pipeline(steps=[
        ('tfidf', text_transformer),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline, text_transformer

# Train the model and save it
def train_and_save_model(data_path, model_path='fake_job_model.pkl'):
    data = pd.read_csv(data_path)
    data = preprocess_data(data)
    
    X = data['description'] + " " + data['requirements']
    y = data['fraudulent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline, vectorizer = build_pipeline()
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Save both model and vectorizer
    joblib.dump(pipeline, model_path)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print(f"Model and vectorizer saved successfully.")
    
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
