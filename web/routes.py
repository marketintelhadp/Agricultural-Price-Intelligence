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
from config import CONFIG
from keras.metrics import MeanSquaredError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_trend_plot(past_prices, future_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(past_prices)), past_prices, label='Past Prices', color='blue')
    plt.plot(range(len(past_prices), len(past_prices) + len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Trend Plot of Past Prices and Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (per kg)')
    plt.legend()
    plt.grid()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

def setup_routes(app):
    @app.route('/')
    def home():
        try:
            markets = sorted(CONFIG.keys())
            selected_market = markets[0] if markets else ''
            fruits = sorted(CONFIG[selected_market].keys()) if selected_market else []
            selected_fruit = fruits[0] if fruits else ''
            varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit else []
            selected_variety = varieties[0] if varieties else ''
            grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety else []
            selected_grade = grades[0] if grades else ''

            return render_template('predict.html',
                                   config=CONFIG,
                                   markets=markets,
                                   fruits=fruits,
                                   varieties=varieties,
                                   grades=grades,
                                   selected_market=selected_market,
                                   selected_fruit=selected_fruit,
                                   selected_variety=selected_variety,
                                   selected_grade=selected_grade,
                                   num_predictions=7)
        except Exception as e:
            logging.error(f"Error rendering template: {str(e)}")
            return "Template not found", 404

    @app.route('/predict_future', methods=['POST'])
    def predict_future():
        try:
            selected_market = request.form.get('market')
            selected_fruit = request.form.get('fruit')
            selected_variety = request.form.get('variety')
            selected_grade = request.form.get('grade')
            num_predictions = int(request.form.get('num_predictions', 7))

            try:
                config_entry = CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]
                model_path = config_entry['model']
                data_path = config_entry['dataset']
            except KeyError as e:
                logging.error(f"Missing configuration entry: {e}")
                return f"Dataset or model not found for: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            df = pd.read_csv(data_path)
            df = df[df['Mask'] == 1]
            df['Date'] = pd.to_datetime(df['Date'])
            max_date = df['Date'].max()
            df = df[df['Date'] >= max_date - pd.DateOffset(months=5)]

            prices = df['Avg Price (per kg)'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(prices)

            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            time_steps = model.input_shape[1]

            if len(prices) < time_steps:
                return jsonify({'error': 'Not enough data to make a prediction.'}), 400

            last_sequence = prices[-time_steps:]
            scaled_sequence = scaler.transform(last_sequence)
            input_sequence = scaled_sequence.reshape(1, time_steps, 1)

            future_predictions = []
            for _ in range(num_predictions):
                prediction = model.predict(input_sequence, verbose=0)
                predicted_price = scaler.inverse_transform(prediction)
                future_predictions.append(float(predicted_price[0][0]))
                input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

            forecast_dates = [(max_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(num_predictions)]
            predicted_prices = zip(forecast_dates, future_predictions)
            past_prices = prices[-30:]
            trend_plot = create_trend_plot(past_prices, future_predictions)

            return render_template('predict.html',
                                   config=CONFIG,
                                   markets=sorted(CONFIG.keys()),
                                   fruits=sorted(CONFIG[selected_market].keys()),
                                   varieties=sorted(CONFIG[selected_market][selected_fruit].keys()),
                                   grades=sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()),
                                   selected_market=selected_market,
                                   selected_fruit=selected_fruit,
                                   selected_variety=selected_variety,
                                   selected_grade=selected_grade,
                                   num_predictions=num_predictions,
                                   predicted_prices=predicted_prices,
                                   trend_plot=trend_plot)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500
