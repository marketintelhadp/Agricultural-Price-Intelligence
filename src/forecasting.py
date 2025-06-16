import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    if len(data) <= seq_length:
        print("Error: Data length must be greater than the sequence length.")
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # Target value

    X, y = np.array(X), np.array(y)
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

# Function to find the best sequence length
def find_best_seq_length(data, max_seq_length):
    best_seq_length = 0
    best_mse = float('inf')
    #for seq_length in range(20, max_seq_length + 1):
    for seq_length in range(10, max_seq_length + 1):
        X, y = create_sequences(data[['Avg Price (per kg)']].values, seq_length)
        if len(X) == 0:
            continue

        temp_model = Sequential([
            LSTM(100, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        temp_model.compile(optimizer='adam', loss='mse')
        temp_model.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=50, batch_size=16, verbose=0)

        predictions = temp_model.predict(X.reshape((X.shape[0], X.shape[1], 1)))
        if np.isnan(predictions).any():
            logging.warning(f"Sequence length {seq_length}: predictions contain NaN, skipping this length.")
            continue

        mse = mean_squared_error(y, predictions)
        if np.isnan(mse):
            logging.warning(f"Sequence length {seq_length}: computed MSE is NaN, skipping this length.")
            continue

        if mse < best_mse:
            best_mse, best_seq_length = mse, seq_length

    return best_seq_length

# Function to train LSTM model
def train_lstm(data, seq_length):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Avg Price (per kg)']].values)

    X, y = create_sequences(data_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(100, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=16, verbose=0, callbacks=[early_stopping])

    return model, scaler

# Function to forecast future values
def forecast_future(model, scaler, data, seq_length, forecast_days):
    data_scaled = scaler.transform(data[['Avg Price (per kg)']].values)
    input_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)

    predictions = []
    for _ in range(forecast_days):
        next_price = model.predict(input_sequence, verbose=0)[0, 0]
        predictions.append(next_price)
        input_sequence = np.append(input_sequence[:, 1:, :], [[[next_price]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

# Function to evaluate model
def evaluate_model(model, X, y, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    return mse, mae

# Save model in new format
def save_model_keras(model, model_path):
    model_path = model_path.replace('.h5', '.keras')
    model.save(model_path, save_format='keras')
    return model_path


import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

def main():
    markets = ["Pulwama"]
    submarket_map = {
        "Pulwama": ["Prichoo"],
        "Shopian": None  # No submarkets
    }

    varieties = ["Kullu Delicious"]
    grades = ["A", "B"]
    forecast_days = 15
    max_seq_length = 40
    results = {}

    for market in markets:
        submarkets = submarket_map.get(market)
        submarkets = submarkets or [None]  # Handle no-submarket case

        for submarket in submarkets:
            base_path = f"data/raw/processed/{market}"
            if submarket:
                base_path = f"{base_path}/{submarket}"

            for variety in varieties:
                # Check if file without grade exists
                no_grade_path = f"{base_path}/{variety}_dataset.csv"
                if os.path.exists(no_grade_path):
                    logging.info(f"Processing {market} {submarket or ''} {variety} (no grade)...")
                    data = pd.read_csv(no_grade_path)
                    if 'Avg Price (per kg)' not in data.columns or 'Mask' not in data.columns:
                        logging.error(f"Missing required columns in {no_grade_path}")
                        continue
                    if data[['Avg Price (per kg)', 'Mask']].isnull().any().any():
                        logging.error(f"NaN values found in {no_grade_path}.")
                        continue
                    data = data[data['Mask'] == 1]
                    if data.empty or data['Avg Price (per kg)'].isnull().any():
                        logging.error(f"Filtered data is empty or contains NaN values for {no_grade_path}.")
                        continue

                    best_seq_length = find_best_seq_length(data, max_seq_length)
                    model, scaler = train_lstm(data, best_seq_length)

                    save_prefix = f"{market}/{submarket}/{variety}" if submarket else f"{market}/{variety}"
                    model_dir = f"models/{save_prefix}"
                    forecast_dir = f"model_forecasts/{save_prefix}"
                    os.makedirs(model_dir, exist_ok=True)
                    os.makedirs(forecast_dir, exist_ok=True)

                    model_path = f"{model_dir}/lstm_{variety}.h5"
                    save_model_keras(model, model_path)

                    predictions = forecast_future(model, scaler, data, best_seq_length, forecast_days)
                    std_dev = np.std(predictions)
                    lower_bound = predictions - 1.96 * std_dev
                    upper_bound = predictions + 1.96 * std_dev

                    key_name = f"{market}_{submarket}_{variety}" if submarket else f"{market}_{variety}"
                    results[key_name] = predictions

                    # Plot
                    plt.plot(range(forecast_days), predictions, label='Forecasted Prices', color='orange')
                    plt.fill_between(range(forecast_days), lower_bound, upper_bound, color='lightgray', alpha=0.5)
                    plt.title(f'Price Forecast: {variety} in {submarket or market}')
                    plt.xlabel('Days')
                    plt.ylabel('Price (per kg)')
                    plt.legend()
                    plt.savefig(f"{forecast_dir}/{variety}_forecast.png")
                    plt.close()

                    forecast_df = pd.DataFrame({
                        'Predictions': predictions,
                        'Lower Bound': lower_bound,
                        'Upper Bound': upper_bound
                    })
                    forecast_df.to_csv(f"{forecast_dir}/{variety}_forecasts.csv", index=False)

                else:
                    for grade in grades:
                        data_path = f"{base_path}/{variety}_{grade}_dataset.csv"
                        if not os.path.exists(data_path):
                            logging.warning(f"File not found: {data_path}")
                            continue

                        logging.info(f"Processing {market} {submarket or ''} {variety} Grade {grade}...")

                        data = pd.read_csv(data_path)
                        if 'Avg Price (per kg)' not in data.columns or 'Mask' not in data.columns:
                            logging.error(f"Missing required columns in {data_path}")
                            continue
                        if data[['Avg Price (per kg)', 'Mask']].isnull().any().any():
                            logging.error(f"NaN values found in {data_path}.")
                            continue
                        data = data[data['Mask'] == 1]
                        if data.empty or data['Avg Price (per kg)'].isnull().any():
                            logging.error(f"Filtered data is empty or contains NaN values for {data_path}.")
                            continue

                        best_seq_length = find_best_seq_length(data, max_seq_length)
                        model, scaler = train_lstm(data, best_seq_length)

                        save_prefix = f"{market}/{submarket}/{variety}/{grade}" if submarket else f"{market}/{variety}/{grade}"
                        model_dir = f"models/{save_prefix}"
                        forecast_dir = f"model_forecasts/{save_prefix}"
                        os.makedirs(model_dir, exist_ok=True)
                        os.makedirs(forecast_dir, exist_ok=True)

                        model_path = f"{model_dir}/lstm_{variety}_grade_{grade}.h5"
                        save_model_keras(model, model_path)

                        predictions = forecast_future(model, scaler, data, best_seq_length, forecast_days)
                        std_dev = np.std(predictions)
                        lower_bound = predictions - 1.96 * std_dev
                        upper_bound = predictions + 1.96 * std_dev

                        key_name = f"{market}_{submarket}_{variety}_grade_{grade}" if submarket else f"{market}_{variety}_grade_{grade}"
                        results[key_name] = predictions

                        # Plot
                        plt.plot(range(forecast_days), predictions, label='Forecasted Prices', color='orange')
                        plt.fill_between(range(forecast_days), lower_bound, upper_bound, color='lightgray', alpha=0.5)
                        plt.title(f'Price Forecast: {variety} Grade {grade} in {submarket or market}')
                        plt.xlabel('Days')
                        plt.ylabel('Price (per kg)')
                        plt.legend()
                        plt.savefig(f"{forecast_dir}/{variety}_grade_{grade}_forecast.png")
                        plt.close()

                        forecast_df = pd.DataFrame({
                            'Predictions': predictions,
                            'Lower Bound': lower_bound,
                            'Upper Bound': upper_bound
                        })
                        forecast_df.to_csv(f"{forecast_dir}/{variety}_Grade_{grade}_forecasts.csv", index=False)

if __name__ == "__main__":
    main()
