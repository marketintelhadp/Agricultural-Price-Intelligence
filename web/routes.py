from flask import request, jsonify, render_template
import numpy as np
import pandas as pd
import logging
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError
from config import CONFIG
from datetime import datetime
import plotly.graph_objs as go
import plotly.io as pio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_forecast_plot(forecast_dates, future_predictions):
    trace = go.Scatter(x=forecast_dates, y=future_predictions, mode='lines+markers', name='Forecast')
    layout = go.Layout(title='Forecasted Prices',
                       xaxis_title='Date',
                       yaxis_title='Price (per kg)',
                       template='plotly_white',
                       margin=dict(l=30, r=30, t=50, b=30))
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

def align_forecast_dates_to_previous_year(df, forecast_days, target_year):
    df = df.copy()
    df["MonthDay"] = df["Date"].dt.strftime("%m-%d")
    unique_md = sorted(df["MonthDay"].unique())
    if len(unique_md) < forecast_days:
        raise ValueError("Not enough date variety in past data for forecast window")
    selected_md = unique_md[:forecast_days]
    forecast_dates = [f"{target_year}-{md}" for md in selected_md]
    return forecast_dates

def setup_routes(app):
    @app.route('/')
    def home():
        try:
            markets = sorted(CONFIG.keys())
            selected_market = request.args.get('market', markets[0] if markets else '')
            fruits = sorted(CONFIG[selected_market].keys()) if selected_market in CONFIG else []
            selected_fruit = request.args.get('fruit', fruits[0] if fruits else '')
            varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit in CONFIG[selected_market] else []
            selected_variety = request.args.get('variety', varieties[0] if varieties else '')
            grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety in CONFIG[selected_market][selected_fruit] else []
            selected_grade = request.args.get('grade', grades[0] if grades else '')

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

            adjusted_market = selected_market
            if selected_market.startswith("Pachhar"):
                adjusted_market = "Pulwama"
                location_key = "Pachhar"
            elif selected_market.startswith("Prichoo"):
                adjusted_market = "Pulwama"
                location_key = "Prichoo"
            else:
                location_key = None

            try:
                if location_key:
                    config_entry = CONFIG[adjusted_market][selected_fruit][location_key][selected_variety][selected_grade]
                else:
                    config_entry = CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]
                model_path = config_entry['model']
                data_path = config_entry['dataset']
            except KeyError as e:
                logging.error(f"Missing configuration entry: {e}")
                return f"Dataset or model not found for: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            df = pd.read_csv(data_path)
            df = df[df['Mask'] == 1]
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)

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

            forecast_dates = align_forecast_dates_to_previous_year(df, num_predictions, target_year=datetime.now().year)
            predicted_prices = list(zip(forecast_dates, future_predictions))
            forecast_plot = create_forecast_plot(forecast_dates, future_predictions)

            markets = sorted(CONFIG.keys())
            fruits = sorted(CONFIG[adjusted_market].keys())
            varieties = sorted(CONFIG[adjusted_market][selected_fruit][location_key].keys()) if location_key else sorted(CONFIG[selected_market][selected_fruit].keys())
            grades = sorted(CONFIG[adjusted_market][selected_fruit][location_key][selected_variety].keys()) if location_key else sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys())

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
                                   num_predictions=num_predictions,
                                   predicted_prices=predicted_prices,
                                   trend_plot=forecast_plot)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500
