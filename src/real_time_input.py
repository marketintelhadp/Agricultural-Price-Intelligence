import os
import pandas as pd
import time
import logging
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to simulate real-time data input
def get_real_time_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"New data loaded from {file_path}")
        return data
    else:
        logging.warning(f"Data file {file_path} not found")
        return None

# Function to process and predict using LSTM model
def process_and_predict(model, data, seq_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Avg Price (per kg)']].values)
    
    # Create sequences
    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
    
    X = np.array(X).reshape((X.shape[0], X.shape[1], 1))
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Main function to simulate real-time input and processing
def main():
    model_path = "models/Pulwama/Prichoo/Delicious/lstm_best_Delicious_grade_A.h5"
    
    if not os.path.exists(model_path):
        logging.error("Model file not found!")
        return
    
    model = load_model(model_path)
    file_path = "data/raw/processed/Pulwama/Prichoo/Delicious_A.csv"
    
    while True:
        data = get_real_time_data(file_path)
        if data is not None:
            predictions = process_and_predict(model, data, seq_length=50)
            logging.info(f"Predictions: {predictions[:5]}")  # Displaying first 5 predictions
        
        time.sleep(60)  # Simulating new data input every 60 seconds

if __name__ == "__main__":
    main()
