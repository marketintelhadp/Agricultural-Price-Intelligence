import os
from keras.metrics import MeanSquaredError

class Config:
    # Model and Data Paths
    MODEL_PATH = r'D:\ML Repositories\Price_forecasting_project\models\Shopian\lstm_Delicious_grade_B.h5'
    DATA_PATH = r'D:\ML Repositories\Price_forecasting_project\data\raw\processed\Shopian\Delicious_B_dataset.csv'

    # Custom objects for loading model
    CUSTOM_OBJECTS = {'mse': MeanSquaredError()}

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key_here')
