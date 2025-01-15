# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from prophet import Prophet
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Load the dataset
data = pd.read_csv(r'D:\ML Repositories\Price_forecasting_project\data\raw\processed\Delicious_A_dataset.csv')

# Ensure proper datetime format for models requiring 'ds'
data = data.rename(columns={"Date": "ds", "Avg Price (per kg)": "y"})
data['ds'] = pd.to_datetime(data['ds'])

# Filter for available data (Mask=1) for SARIMA and Prophet
available_data = data[data['Mask'] == 1].copy()
available_data.reset_index(inplace=True)

# Split data for training and testing
train_data = available_data[available_data['ds'] < '2024-09-15']
test_data = available_data[available_data['ds'] >= '2024-09-15']

# Function to reverse scaling
def reverse_scaling(scaled_values, data):
    return scaled_values * (data['y'].max() - data['y'].min()) + data['y'].min()

# SARIMA Model
def sarima_model(train_data, test_data):
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 3)
    D_values = range(0, 2)
    Q_values = range(0, 3)
    seasonal_periods = [12]

    param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods))
    results = []

    for param in param_grid:
        try:
            p, d, q, P, D, Q, S = param
            sarima_model = SARIMAX(train_data['y'], order=(p, d, q), seasonal_order=(P, D, Q, S))
            sarima_results = sarima_model.fit(disp=False)
            results.append((param, sarima_results.aic))
        except Exception as e:
            continue

    if results:
        results_df = pd.DataFrame(results, columns=["Params", "AIC"])
        best_params = results_df.sort_values(by="AIC").iloc[0]
        print("\nBest Parameters for SARIMA:", best_params)

        best_p, best_d, best_q, best_P, best_D, best_Q, best_S = best_params[0]
        sarima_model = SARIMAX(train_data['y'], order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, best_S))
        sarima_results = sarima_model.fit(disp=False)
        sarima_forecast = sarima_results.get_forecast(steps=len(test_data))
        return sarima_forecast.predicted_mean
    else:
        print("No valid SARIMA models were successfully fitted.")
        return None

# Prophet Model
def prophet_model(train_data, test_data):
    prophet_model = Prophet()
    prophet_model.fit(train_data[['ds', 'y']])
    future = prophet_model.make_future_dataframe(periods=len(test_data))
    prophet_forecast = prophet_model.predict(future)
    return prophet_forecast['yhat'][-len(test_data):]

# Function to create lagged features
def create_lagged_features(data, max_lag):
    for lag in range(1, max_lag + 1):
        data[f'y_lag{lag}'] = data['y'].shift(lag)
    return data

# Function to find optimal lags
def find_best_lags(model, train_data, val_data, features, max_lag):
    best_lags, best_mse = 0, float('inf')
    for num_lags in range(1, max_lag + 1):
        lag_features = features[:num_lags]
        model.fit(train_data[lag_features], train_data['y'])
        val_predictions = model.predict(val_data[lag_features])
        ```python
        mse = mean_squared_error(val_data['y'], val_predictions)
        if mse < best_mse:
            best_mse, best_lags = mse, num_lags
    return best_lags, best_mse

# Random Forest Model
def random_forest_model(train_data, test_data):
    max_lag = 60
    train_data = create_lagged_features(train_data, max_lag)
    test_data = create_lagged_features(test_data, max_lag)

    train_subset = train_data[train_data['ds'] < '2023-09-15']
    val_subset = train_data[(train_data['ds'] >= '2023-09-15') & (train_data['ds'] < '2024-09-15')]

    features = [f'y_lag{i}' for i in range(1, max_lag + 1)]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_lags_rf, _ = find_best_lags(rf_model, train_subset, val_subset, features, max_lag)

    final_rf_features = features[:best_lags_rf]
    rf_model.fit(train_data[final_rf_features], train_data['y'])
    rf_predictions = rf_model.predict(test_data[final_rf_features])
    return rf_predictions

# XGBoost Model
def xgboost_model(train_data, test_data):
    max_lag = 60
    train_data = create_lagged_features(train_data, max_lag)
    test_data = create_lagged_features(test_data, max_lag)

    train_subset = train_data[train_data['ds'] < '2023-09-15']
    val_subset = train_data[(train_data['ds'] >= '2023-09-15') & (train_data['ds'] < '2024-09-15')]

    features = [f'y_lag{i}' for i in range(1, max_lag + 1)]
    scaler = StandardScaler()
    train_data['y_scaled'] = scaler.fit_transform(train_data[['y']])
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    best_lags_xgb, _ = find_best_lags(xgb_model, train_subset, val_subset, features, max_lag)

    final_xgb_features = features[:best_lags_xgb]
    xgb_model.fit(train_data[final_xgb_features], train_data['y_scaled'])
    xgb_predictions_scaled = xgb_model.predict(test_data[final_xgb_features])
    xgb_predictions = scaler.inverse_transform(xgb_predictions_scaled.reshape(-1, 1))
    return xgb_predictions

# LSTM Model
def lstm_model(train_data, test_data):
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    max_seq_length = 10  # Example sequence length
    train_scaled = MinMaxScaler().fit_transform(train_data[['y']])
    test_scaled = MinMaxScaler().fit_transform(test_data[['y']])

    X_train, y_train = create_sequences(train_scaled, max_seq_length)
    X_test, y_test = create_sequences(test_scaled, max_seq_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = Sequential([
        LSTM(100, activation='relu', input_shape=(max_seq_length, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    lstm_pred = lstm_model.predict(X_test)
    return lstm_pred.flatten()

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return self.fc_out(x)

def transformer_model(train_data ```python
, test_data):
    max_seq_length = 10  # Example sequence length
    train_scaled = MinMaxScaler().fit_transform(train_data[['y']])
    test_scaled = MinMaxScaler().fit_transform(test_data[['y']])

    X_train, y_train = create_sequences(train_scaled, max_seq_length)
    X_test, y_test = create_sequences(test_scaled, max_seq_length)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    transformer = TransformerModel(input_dim=1, embed_dim=16, num_heads=2, ff_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

    for epoch in range(100):
        transformer.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = transformer(batch_X.unsqueeze(-1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    transformer.eval()
    with torch.no_grad():
        transformer_pred = transformer(X_test.unsqueeze(-1)).numpy()
    
    return reverse_scaling(transformer_pred, test_data)

# Main function to run all models
def main():
    sarima_pred = sarima_model(train_data, test_data)
    prophet_pred = prophet_model(train_data, test_data)
    rf_pred = random_forest_model(train_data, test_data)
    xgb_pred = xgboost_model(train_data, test_data)
    lstm_pred = lstm_model(train_data, test_data)
    transformer_pred = transformer_model(train_data, test_data)

    # Calculate and print metrics for each model
    metrics = {
        'SARIMA': mean_squared_error(test_data['y'], sarima_pred),
        'Prophet': mean_squared_error(test_data['y'], prophet_pred),
        'Random Forest': mean_squared_error(test_data['y'], rf_pred),
        'XGBoost': mean_squared_error(test_data['y'], xgb_pred),
        'LSTM': mean_squared_error(test_data['y'], lstm_pred),
        'Transformer': mean_squared_error(test_data['y'], transformer_pred)
    }

    for model, mse in metrics.items():
        print(f"{model} - Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()