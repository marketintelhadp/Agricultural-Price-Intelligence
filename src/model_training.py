import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load the dataset and preprocess
def load_data(file_path):
    """
    Load and preprocess the dataset.
    """
    data = pd.read_csv(file_path)
    data = data.rename(columns={"Date": "ds", "Avg Price (per kg)": "y"})
    data['ds'] = pd.to_datetime(data['ds'])
    return data

# Split data into training and testing
def split_data(data, split_date):
    """
    Split data into training and testing based on a date.
    """
    train = data[data['ds'] < split_date]
    test = data[data['ds'] >= split_date]
    return train, test

# Function to create lagged features
def create_lagged_features(data, max_lag):
    """
    Create lagged features based on max_lag.
    """
    for lag in range(1, max_lag + 1):
        data[f'y_lag{lag}'] = data['y'].shift(lag)
    return data

# SARIMA Model tuning
def sarima_tuning(train_data, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods):
    """
    Perform grid search to find the best SARIMA parameters.
    """
    param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods))
    results = []
    
    for param in param_grid:
        try:
            p, d, q, P, D, Q, S = param
            model = SARIMAX(train_data['y'], order=(p, d, q), seasonal_order=(P, D, Q, S),
                            enforce_stationarity=False, enforce_invertibility=False)
            results_fit = model.fit(disp=False)
            results.append((param, results_fit.aic))
        except Exception as e:
            print(f"Error with parameters {param}: {e}")
            continue

    if results:
        results_df = pd.DataFrame(results, columns=["Params", "AIC"])
        best_params = results_df.sort_values(by="AIC").iloc[0]
        return best_params
    else:
        raise ValueError("No valid SARIMA models were successfully fitted.")

# Function to evaluate SARIMA Model
def train_sarima(train_data, test_data, params):
    """
    Train and forecast using the SARIMA model.
    """
    p, d, q, P, D, Q, S = params
    model = SARIMAX(train_data['y'], order=(p, d, q), seasonal_order=(P, D, Q, S))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=len(test_data))
    pred = forecast.predicted_mean
    mse = mean_squared_error(test_data['y'], pred)
    mae = mean_absolute_error(test_data['y'], pred)
    return pred, mse, mae

# Random Forest Model
def random_forest_model(train_data, val_subset, test_data, features):
    """
    Train Random Forest Model and make predictions.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_data[features], train_data['y'])
    rf_predictions = rf_model.predict(test_data[features])
    mse = mean_squared_error(test_data['y'], rf_predictions)
    mae = mean_absolute_error(test_data['y'], rf_predictions)
    return mse, mae, rf_predictions

# XGBoost Model
def xgboost_model(train_data, val_subset, test_data, features):
    """
    Train XGBoost Model and make predictions.
    """
    scaler = StandardScaler()
    train_data['y_scaled'] = scaler.fit_transform(train_data[['y']])
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(train_data[features], train_data['y_scaled'])
    xgb_predictions_scaled = xgb_model.predict(test_data[features])
    xgb_predictions = scaler.inverse_transform(xgb_predictions_scaled.reshape(-1, 1))
    mse = mean_squared_error(test_data['y'], xgb_predictions)
    mae = mean_absolute_error(test_data['y'], xgb_predictions)
    return mse, mae, xgb_predictions

# Prophet Model
def prophet_model(train_data, test_data):
    """
    Train Prophet Model and make predictions.
    """
    prophet = Prophet()
    prophet.fit(train_data[['ds', 'y']])
    future = prophet.make_future_dataframe(periods=len(test_data))
    forecast = prophet.predict(future)
    predictions = forecast['yhat'][-len(test_data):]
    mse = mean_squared_error(test_data['y'], predictions)
    mae = mean_absolute_error(test_data['y'], predictions)
    return mse, mae, predictions

# LSTM Model
def lstm_model(train_data, test_data, seq_length):
    """
    Train LSTM model and make predictions.
    """
    X_train, y_train = create_sequences(train_data['y'].values, seq_length)
    X_test, y_test = create_sequences(test_data['y'].values, seq_length)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    lstm_model = Sequential([LSTM(100, activation='relu', input_shape=(seq_length, 1)),
                             Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    lstm_predictions = lstm_model.predict(X_test)
    mse = mean_squared_error(y_test, lstm_predictions)
    mae = mean_absolute_error(y_test, lstm_predictions)
    return mse, mae, lstm_predictions

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc_out(x[:, -1, :])  # We take the last output as prediction
        return x

def transformer_model(train_data, test_data, seq_length):
    """
    Train Transformer model and make predictions.
    """
    X_train, y_train = create_sequences(train_data['y'].values, seq_length)
    X_test, y_test = create_sequences(test_data['y'].values, seq_length)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    model = TransformerModel(input_dim=1, output_dim=1, num_heads=4, num_layers=2, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    mse = mean_squared_error(y_test, predictions.numpy())
    mae = mean_absolute_error(y_test, predictions.numpy())
    return mse, mae, predictions.numpy()

# Main function to train models for selected varieties and grades
def train_models_for_variety_and_grade(file_path, split_date, varieties, grades, seq_length=10):
    data = load_data(file_path)
    
    for variety in varieties:
        for grade in grades:
            print(f"Training models for Variety: {variety}, Grade: {grade}")
            # Filter data based on variety and grade
            variety_grade_data = data[(data['Variety'] == variety) & (data['Grade'] == grade)]
            
            # Split data into train and test
            train_data, test_data = split_data(variety_grade_data, split_date)
            
            # SARIMA tuning and training
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            P_values = range(0, 3)
            D_values = range(0, 2)
            Q_values = range(0, 3)
            seasonal_periods = [7]
            best_params = sarima_tuning(train_data, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods)
            sarima_pred, sarima_mse, sarima_mae = train_sarima(train_data, test_data, best_params)
            
            # Random Forest Model
            features = ['feature1', 'feature2']  # Example features
            rf_mse, rf_mae, rf_predictions = random_forest_model(train_data, None, test_data, features)
            
            # XGBoost Model
            xgb_mse, xgb_mae, xgb_predictions = xgboost_model(train_data, None, test_data, features)
            
            # Prophet Model
            prophet_mse, prophet_mae, prophet_predictions = prophet_model(train_data, test_data)
            
            # LSTM Model
            lstm_mse, lstm_mae, lstm_predictions = lstm_model(train_data, test_data, seq_length)
            
            # Transformer Model
            transformer_mse, transformer_mae, transformer_predictions = transformer_model(train_data, test_data, seq_length)
            
            # Store or print results as needed
            print(f"SARIMA MSE: {sarima_mse}, MAE: {sarima_mae}")
            print(f"Random Forest MSE: {rf_mse}, MAE: {rf_mae}")
            print(f"XGBoost MSE: {xgb_mse}, MAE: {xgb_mae}")
            print(f"Prophet MSE: {prophet_mse}, MAE: {prophet_mae}")
            print(f"LSTM MSE: {lstm_mse}, MAE: {lstm_mae}")
            print(f"Transformer MSE: {transformer_mse}, MAE: {transformer_mae}")

# Example usage:
file_path = 'path_to_your_data.csv'
varieties = ['Variety1', 'Variety2']
grades = ['Grade1', 'Grade2']
train_models_for_variety_and_grade(file_path, '2022-01-01', varieties, grades)