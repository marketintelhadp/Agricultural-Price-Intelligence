from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model with custom loss function
model_path = r'models/Shopian/lstm_Delicious_grade_A.h5'
custom_objects = {'mse': MeanSquaredError()}
model = load_model(model_path, custom_objects=custom_objects)

# Load the dataset and fit the scaler
data_path = r'D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\raw\processed\Shopian\Delicious_A_dataset.csv'
data = pd.read_csv(data_path)

# Assuming the relevant time series data is in a column named 'Avg Price (per kg)'
prices = data['Avg Price (per kg)'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler()
scaler.fit(prices)

# Get the input shape to determine the number of time steps
input_shape = model.input_shape
time_steps = input_shape[1]  # Number of time steps
logging.info(f"Input Shape: {input_shape}")

@app.route('/')
def home():
    return """
    <h1>LSTM Price Prediction API</h1>
    <p>This API allows you to make predictions on future prices.</p>
    <p>Use the <code>/predict_future</code> endpoint to get future price predictions.</p>
    <form action="/predict_future" method="post">
        <label for="num_predictions">Number of Predictions:</label>
        <input type="number" id="num_predictions" name="num_predictions" required>
        <input type="submit" value="Get Predictions">
    </form>
    """

@app.route('/predict_future', methods=['POST'])
def predict_future():
    try:
        num_predictions = int(request.form['num_predictions'])
        
        # Ensure there is enough data to make predictions
        if len(prices) < time_steps:
            return jsonify({'error': 'Not enough data to make a prediction. Please provide more data.'}), 400

        # Use the last 'time_steps' data points
        last_sequence = prices[-time_steps:].reshape(-1, 1)
        scaled_sequence = scaler.transform(last_sequence)
        input_sequence = scaled_sequence.reshape(1, time_steps, 1)

        future_predictions = []
        for _ in range(num_predictions):
            prediction = model.predict(input_sequence)
            predicted_price = scaler.inverse_transform(prediction)
            future_predictions.append(float(predicted_price[0][0]))

            # Update the input sequence with the new prediction
            input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

        logging.info(f"Future predictions: {future_predictions}")
        return jsonify({'predicted_prices': future_predictions})

    except Exception as e:
        logging.error(f"Error during future prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Set the host and port for deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
