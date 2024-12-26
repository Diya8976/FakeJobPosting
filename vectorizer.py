import pandas as pd
import joblib
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Preprocess the dataset
def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values and clean text fields.
    
    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Drop columns with too many missing values (optional)
    df = df.drop(columns=['salary_range', 'department'], errors='ignore')
    
    # Fill missing values for text fields
    text_fields = ['description', 'company_profile', 'requirements', 'benefits']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('').str.lower()
            df[field] = df[field].str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
    
    print("Data preprocessing completed.")
    return df

# Build the ML pipeline
def build_pipeline():
    """
    Build and return the ML pipeline for text processing and classification.
    
    Returns:
        sklearn.pipeline.Pipeline: Configured pipeline.
    """
    # Text vectorization with TF-IDF
    text_transformer = Pipeline(steps=[ 
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
    ])
    
    # Preprocessing pipeline for columns
    preprocessor = ColumnTransformer(transformers=[
        ('description', text_transformer, 'description'),
        ('company_profile', text_transformer, 'company_profile'),
        ('requirements', text_transformer, 'requirements')
    ], remainder='passthrough')  # Keep other columns
    
    # Full pipeline with a classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Pipeline created.")
    return pipeline

# Train the model and save it
def train_and_save_model(data_path, model_path, vectorizer_path):
    """
    Train the ML pipeline on the dataset and save the trained model and vectorizer.
    
    Args:
        data_path (str): Path to the input dataset (CSV file).
        model_path (str): Path to save the trained model.
        vectorizer_path (str): Path to save the vectorizer.
    """
    # Load dataset
    data = pd.read_csv(data_path)
    print("Dataset loaded.")
    
    # Preprocess dataset
    data = preprocess_data(data)
    
    # Features and target
    X = data[['description', 'company_profile', 'requirements']]
    y = data['fraudulent'] if 'fraudulent' in data.columns else None
    
    # Handle missing target
    if y is None:
        print("Target column 'fraudulent' not found. Exiting...")
        return
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
    
    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Model training completed.")
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save the trained pipeline and vectorizer
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps['preprocessor'].named_transformers_['description'].named_steps['tfidf'], vectorizer_path)
    
    print(f"Model saved at {model_path}")
    print(f"Vectorizer saved at {vectorizer_path}")

# Run the script
if __name__ == "__main__":
    # Set paths for extracted dataset and saving model/vectorizer
    extracted_path = 'C:/Users/diya/Desktop/jupyter projects/fake_job_posts/fake_job_postings.csv/fake_job_postings.csv'
    model_path = 'C:/Users/diya/Desktop/jupyter projects/fake_job_posts/fake_job_model.pkl'
    vectorizer_path = 'C:/Users/diya/Desktop/jupyter projects/fake_job_posts/vectorizer.pkl'
    
    train_and_save_model(extracted_path, model_path, vectorizer_path)