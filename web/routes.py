from flask import request, jsonify, render_template
import numpy as np
import pandas as pd
import logging
import os
import io
import base64
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load LSTM model
model = load_model(Config.MODEL_PATH, custom_objects=Config.CUSTOM_OBJECTS)

# Load dataset
data = pd.read_csv(Config.DATA_PATH)
data = data[data['Mask'] == 1]

# Extract relevant price data
prices = data['Avg Price (per kg)'].values.reshape(-1, 1)

# Initialize and fit scaler
scaler = MinMaxScaler()
scaler.fit(prices)

# Get model input shape for time steps
input_shape = model.input_shape
time_steps = input_shape[1]  
logging.info(f"Model Input Shape: {input_shape}")

def create_trend_plot(past_prices, future_predictions):
    """Generate a trend plot of past prices and future predictions."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(past_prices)), past_prices, label='Past Prices', color='blue')
    plt.plot(range(len(past_prices), len(past_prices) + len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Trend Plot of Past Prices and Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (per kg)')
    plt.legend()
    plt.grid()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

def setup_routes(app):
    """Register routes for the Flask app."""

    @app.route('/')
    def home():
        try:
            return render_template('predict.html')
        except Exception as e:
            logging.error(f"Error rendering template: {str(e)}")
            return "Template not found", 404

    @app.route('/predict_future', methods=['POST'])
    def predict_future():
        try:
            num_predictions = int(request.form['num_predictions'])

            if len(prices) < time_steps:
                return jsonify({'error': 'Not enough data to make a prediction. Please provide more data.'}), 400

            # Use last 'time_steps' data points
            last_sequence = prices[-time_steps:].reshape(-1, 1)
            scaled_sequence = scaler.transform(last_sequence)
            input_sequence = scaled_sequence.reshape(1, time_steps, 1)

            future_predictions = []
            for _ in range(num_predictions):
                prediction = model.predict(input_sequence)
                predicted_price = scaler.inverse_transform(prediction)
                future_predictions.append(float(predicted_price[0][0]))

                # Update input sequence with new prediction
                input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

            logging.info(f"Future predictions: {future_predictions}")

            # Get last 30 prices for trend plot
            past_prices = prices[-30:]
            trend_plot = create_trend_plot(past_prices, future_predictions)

            return render_template('predict.html', predicted_prices=future_predictions, trend_plot=trend_plot)

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500
