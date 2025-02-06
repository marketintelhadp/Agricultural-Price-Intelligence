from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import matplotlib.pyplot as plt
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__, template_folder='static/templates')

# Load the model with custom loss function
model_path = 'models/Shopian/lstm_Delicious_grade_A.h5'
custom_objects = {'mse': MeanSquaredError()}
model = load_model(model_path, custom_objects=custom_objects)

# Load the dataset and fit the scaler
data_path = 'D:/Git Projects/Price_forecasting_project/Agricultural-Price-Intelligence/data/raw/processed/Shopian/Delicious_A_dataset.csv'
data = pd.read_csv(data_path)
data = data[data['Mask'] == 1]

# Assuming the relevant time series data is in a column named 'Avg Price (per kg)'
prices = data['Avg Price (per kg)'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler()
scaler.fit(prices)

# Get the input shape to determine the number of time steps
input_shape = model.input_shape
time_steps = input_shape[1]  # Number of time steps
logging.info(f"Input Shape: {input_shape}")

def create_trend_plot(past_prices, future_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(past_prices)), past_prices, label='Past Prices', color='blue')
    plt.plot(range(len(past_prices), len(past_prices) + len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Trend Plot of Past Prices and Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (per kg)')
    plt.legend()
    plt.grid()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

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

        # Create the trend plot
        # Get the last 30 prices and inverse transform them
        past_prices = prices[-30:]  # Get the last 30 prices
        #past_prices = scaler.inverse_transform(past_prices_scaled)  # Inverse transform to original scale
        trend_plot = create_trend_plot(past_prices, future_predictions)

        return render_template('predict.html', predicted_prices=future_predictions, trend_plot=trend_plot)

    except Exception as e:
        logging.error(f"Error during future prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Set the host and port for deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT ', 500)), debug=True)