from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Correct file for vectorizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    description = request.form['description']
    company_profile = request.form['company_profile']
    requirements = request.form['requirements']

    # Prepare data for prediction (same as during training)
    input_data = pd.DataFrame([{
        'description': description,
        'company_profile': company_profile,
        'requirements': requirements
    }])
    input_data['combined_text'] = input_data['description'] + " " + input_data['requirements']
    
    # Transform input with vectorizer
    transformed_input = vectorizer.transform(input_data['combined_text'])

    # Make prediction
    prediction = model.predict(transformed_input)
    result = "Fake" if prediction[0] == 1 else "Legitimate"

    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
