import os
from keras.metrics import MeanSquaredError

class Config:
    # Model and Data Paths
    MODEL_PATH = r'D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\models\Narwal\Delicious\lstm_Delicious.h5'
    DATA_PATH = r'D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\raw\processed\Narwal\Delicious_dataset.csv'

    # Custom objects for loading model
    CUSTOM_OBJECTS = {'mse': MeanSquaredError()}

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key_here')
