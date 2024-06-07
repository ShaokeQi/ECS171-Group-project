from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LinearRegression
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
with open('LinearRegression_model.pkl', 'rb') as f:
    linear_regression_model = pickle.load(f)

with open('Perceptron_model.pkl', 'rb') as f:
    perceptron_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

dfnn_model = load_model('DFNN_best_model.h5', compile=False)

# Prediction function
def predict(input_features, model):
    input_features = scaler.transform([input_features])
    if isinstance(model, (LinearRegression, Perceptron)):
        prediction = model.predict(input_features)
        # Perceptron model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_features)
            positive_class_probability = probabilities[0][1]
            return prediction[0], positive_class_probability
        else:
            return prediction[0], None
    else:
        # Keras model
        prediction = model.predict(input_features)
        return prediction[0][0], None

# Calculate accuracy
def calculate_model_accuracy(model, X_test, y_test):
    test_pred = model.predict(X_test)
    test_pred = (test_pred >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, test_pred)
    return accuracy

# Calculate and cache model accuracies
diabetes_data = pd.read_csv('diabetes.csv')
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

linear_regression_accuracy = calculate_model_accuracy(linear_regression_model, X_test, y_test)
perceptron_accuracy = calculate_model_accuracy(perceptron_model, X_test, y_test)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        features = [float(request.form[key]) for key in request.form.keys() if key != 'model']
        model_choice = request.form['model']

        if model_choice == 'Linear Regression':
            prediction, probability = predict(features, linear_regression_model)
            prediction_class = 1 if prediction >= 0.5 else 0
            accuracy = linear_regression_accuracy
            return render_template('index.html',
                                   prediction_text=f'Prediction: {prediction}, Diabetes (1: true, 0: false): {prediction_class}',
                                   model_name='Linear Regression',
                                   accuracy=accuracy)
        elif model_choice == 'Perceptron':
            prediction, _ = predict(features, perceptron_model)
            accuracy = perceptron_accuracy
            return render_template('index.html', prediction_text=f'Diabetes Prediction (1: true, 0: false): {prediction}',
                                   model_name='Perceptron',
                                   accuracy=accuracy)
        elif model_choice == 'DFNN':
            prediction, _ = predict(features, dfnn_model)
            prediction_class = 1 if prediction >= 0.5 else 0
            return render_template('index.html',
                                   prediction_text=f'Diabetes Prediction (1: true, 0: false): {prediction_class}',
                                   model_name='DFNN')
        else:
            return render_template('index.html', error_text='Invalid model selection.')
    except Exception as e:
        return render_template('index.html', error_text=f'Error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)




