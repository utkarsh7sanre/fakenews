from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model from the pickle file

# Load your pre-trained model and vectorizer from the joblib files
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')


@app.route('/')
def index():
    return render_template('index.html')  # Create an HTML form for user input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = request.form['user_input']

        # Preprocess user input (clean, split, vectorize, etc.) as needed

        # Vectorize the user input using the same TF-IDF vectorizer used during training
        user_input_vectorized = vectorizer.transform([user_input])

        # Make a prediction using the pre-trained logistic regression model
        prediction = model.predict(user_input_vectorized)

        # Assuming 0 means "Not Fake" and 1 means "Fake", you can convert it to a human-readable label
        result = "Fake" if prediction[0] == 1 else "Not Fake"

        return render_template('result.html', result=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
