import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load and clean data
def load_and_clean_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")
        data = data.dropna(subset=['Avg Price (per kg)', 'Mask'])
        return data
    else:
        logging.warning(f"Data file {file_path} not found!")
        return None

# Function to scale data
def scale_data(data, column_name='Avg Price (per kg)'):
    if column_name in data.columns:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[column_name]].values)
        logging.info(f"Data scaled using {column_name}")
        return scaled_data, scaler
    else:
        logging.error(f"Column {column_name} not found in data.")
        return None, None

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # Target value
    
    return np.array(X), np.array(y)

# Function to save predictions to a file
def save_predictions(predictions, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, predictions, delimiter=',')
    logging.info(f"Predictions saved to {output_path}")

# Function to log model performance
def log_model_performance(mse, mae, r2):
    logging.info(f"Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
