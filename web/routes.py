from flask import request, jsonify, render_template, flash
import numpy as np
import pandas as pd
import logging
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError
from config import CONFIG
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from glob import glob
import os
import json
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_forecast_plot(forecast_dates, future_predictions):
    trace = go.Scatter(x=forecast_dates, y=future_predictions, mode='lines+markers', name='Forecast')
    layout = go.Layout(title='Forecasted Prices', xaxis_title='Date', yaxis_title='Price (per kg)',
                       template='plotly_white', margin=dict(l=30, r=30, t=50, b=30))
    return pio.to_html(go.Figure(data=[trace], layout=layout), full_html=False)

def align_forecast_dates_to_previous_year(df, forecast_days, target_year):
    df = df.copy()
    df['MonthDay'] = df['Date'].dt.strftime('%m-%d')
    unique_md = sorted(df['MonthDay'].unique())
    if len(unique_md) < forecast_days:
        raise ValueError("Not enough date variety in past data for forecast window")
    return [f"{target_year}-{md}" for md in unique_md[:forecast_days]]

def create_dashboard_plot(df):
    import plotly.graph_objs as go
    import plotly.io as pio
    import json

    # Filter only actual sales
    df = df[df['Mask'] == 1].copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a Plotly scatter line chart without connecting missing dates
    trace = go.Scatter(
        x=df['Date'],
        y=df['Price (₹/kg)'],
        mode='lines+markers',
        line=dict(color='orange'),
        marker=dict(size=6),
        name='Actual Sales',
        connectgaps=False  # This is the key fix
    )

    layout = go.Layout(
        title='Recent Price Trends (Only Actual Sales)',
        xaxis_title='Date',
        yaxis_title='Price (₹/kg)',
        template='plotly_white',
        margin=dict(l=40, r=30, t=50, b=40)
    )

    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

def parse_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[df['Mask'] == 1]
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = df['Avg Price (per kg)']
        df.rename(columns={'Avg Price (per kg)': 'Price (₹/kg)'}, inplace=True)

        parts = file_path.split(os.sep)
        market = parts[-2]
        file_name = os.path.basename(file_path).replace('_dataset.csv', '')
        tokens = file_name.split('_')

        if len(tokens) == 2:
            variety, grade = tokens
        elif len(tokens) == 3:
            variety = f"{tokens[0]} {tokens[1]}"
            grade = tokens[2]
        else:
            return None

        fruit = 'Cherry' if 'cherry' in file_path.lower() else 'Apple' if 'apple' in file_path.lower() else 'Unknown'
        df['Market'], df['Fruit'], df['Variety'], df['Grade'] = market, fruit, variety, grade

        return df[['Date', 'Market', 'Fruit', 'Variety', 'Grade', 'Price (₹/kg)', 'Price']]
    except Exception as e:
        logging.warning(f"Skipping file {file_path} due to error: {e}")
        return None

def get_config_options(selected_market, selected_fruit=None, selected_variety=None):
    fruits = sorted(CONFIG[selected_market].keys()) if selected_market in CONFIG else []
    varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_market in CONFIG and selected_fruit in CONFIG[selected_market] else []
    grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_market in CONFIG and selected_fruit in CONFIG[selected_market] and selected_variety in CONFIG[selected_market][selected_fruit] else []
    return fruits, varieties, grades

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
            logging.error(f"Error rendering template: {e}")
            return "Template not found", 404

    @app.route('/predict_future', methods=['POST'])
    def predict_future():
        try:
            selected_market = request.form.get('market')
            selected_fruit = request.form.get('fruit')
            selected_variety = request.form.get('variety')
            selected_grade = request.form.get('grade')
            num_predictions = int(request.form.get('num_predictions', 7))

            adjusted_market, location_key = selected_market, None
            if selected_market.startswith('Pachhar'):
                adjusted_market, location_key = 'Pulwama', 'Pachhar'
            elif selected_market.startswith('Prichoo'):
                adjusted_market, location_key = 'Pulwama', 'Prichoo'

            try:
                config_entry = CONFIG[adjusted_market][selected_fruit][location_key][selected_variety][selected_grade] if location_key else CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]
                model_path = config_entry['model']
                data_path = config_entry['dataset']
            except KeyError as e:
                logging.error(f"Missing config entry: {e}")
                return f"Dataset or model not found for: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            df = pd.read_csv(data_path)
            df = df[df['Mask'] == 1]
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)

            prices = df['Avg Price (per kg)'].values.reshape(-1, 1)
            scaler = MinMaxScaler().fit(prices)
            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            time_steps = model.input_shape[1]

            if len(prices) < time_steps:
                return jsonify({'error': 'Not enough data to make a prediction.'}), 400

            input_sequence = scaler.transform(prices[-time_steps:]).reshape(1, time_steps, 1)
            future_predictions = []
            for _ in range(num_predictions):
                prediction = model.predict(input_sequence, verbose=0)
                predicted_price = scaler.inverse_transform(prediction)
                future_predictions.append(float(predicted_price[0][0]))
                input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

            forecast_dates = align_forecast_dates_to_previous_year(df, num_predictions, datetime.now().year)
            forecast_plot = create_forecast_plot(forecast_dates, future_predictions)
            predicted_prices = list(zip(forecast_dates, future_predictions))

            fruits, varieties, grades = get_config_options(selected_market, selected_fruit, selected_variety)
            return render_template('predict.html', config=CONFIG, markets=sorted(CONFIG.keys()), fruits=fruits,
                                   varieties=varieties, grades=grades,
                                   selected_market=selected_market, selected_fruit=selected_fruit,
                                   selected_variety=selected_variety, selected_grade=selected_grade,
                                   num_predictions=num_predictions, predicted_prices=predicted_prices,
                                   trend_plot=forecast_plot)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return jsonify({'error': 'Prediction failed.'}), 500

    @app.route('/dashboard')
    def dashboard():
        try:
            # Reconstruct CONFIG for dashboard dropdown rendering (preserve original for prediction)
            dashboard_config = {}
            for market, fruits in CONFIG.items():
                if market == "Pulwama":
                    for fruit, submarkets in fruits.items():
                        for submarket, grades in submarkets.items():
                            new_market_key = f"{submarket} Pulwama"
                            if new_market_key not in dashboard_config:
                                dashboard_config[new_market_key] = {}
                            dashboard_config[new_market_key][fruit] = grades
                else:
                    dashboard_config[market] = fruits

            markets = sorted(dashboard_config.keys())
            selected_market = request.args.get('market') or markets[0] if markets else ''

            # Map Pulwama submarkets back
            adjusted_market = "Pulwama"
            location_key = None
            if selected_market.startswith("Pachhar"):
                location_key = "Pachhar"
            elif selected_market.startswith("Prichoo"):
                location_key = "Prichoo"
            else:
                adjusted_market = selected_market

            fruits = sorted(dashboard_config[selected_market].keys()) if selected_market in dashboard_config else []
            selected_fruit = request.args.get('fruit') or (fruits[0] if fruits else '')
            varieties = []
            if location_key:
                varieties = sorted(CONFIG[adjusted_market][selected_fruit][location_key].keys()) if selected_fruit in CONFIG[adjusted_market] and location_key in CONFIG[adjusted_market][selected_fruit] else []
            else:
                varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit in CONFIG[selected_market] else []

            selected_variety = request.args.get('variety') or (varieties[0] if varieties else '')
            grades = []
            if location_key:
                grades = sorted(CONFIG[adjusted_market][selected_fruit][location_key][selected_variety].keys()) if selected_variety in CONFIG[adjusted_market][selected_fruit][location_key] else []
            else:
                grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety in CONFIG[selected_market][selected_fruit] else []

            selected_grade = request.args.get('grade') or (grades[0] if grades else '')

            cards = [
                {'title': 'Selected Market', 'value': selected_market or 'N/A'},
                {'title': 'Selected Fruit', 'value': selected_fruit or 'N/A'},
                {'title': 'Selected Variety', 'value': selected_variety or 'N/A'},
                {'title': 'Selected Grade', 'value': selected_grade or 'N/A'}
            ]

            data = []
            plot_json = '[]'

            try:
                if location_key:
                    config_entry = CONFIG[adjusted_market][selected_fruit][location_key][selected_variety][selected_grade]
                else:
                    config_entry = CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]

                data_path = config_entry['dataset']
                df = pd.read_csv(data_path)
                df = df[df['Mask'] == 1]
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values(by='Date', inplace=True)
                df['Price'] = df['Avg Price (per kg)']
                df.rename(columns={'Avg Price (per kg)': 'Price (₹/kg)'}, inplace=True)

                data = df.tail(150).to_dict(orient='records')
                if df.empty or 'Price (₹/kg)' not in df.columns:
                    flash("No data available for the selected combination.", "warning")
                    return render_template("dashboard.html", config=dashboard_config, data=[], plot_data="", selected_market=selected_market, selected_fruit=selected_fruit, selected_variety=selected_variety, selected_grade=selected_grade, cards=cards)
                plot_json = create_dashboard_plot(df)

            except Exception as e:
                logging.warning(f"No data available for the selected combination: {e}")
                flash("No data available for the selected combination.", "warning")

            plot_img = create_dashboard_plot(df.tail(100))
            return render_template("dashboard.html", config=dashboard_config, data=df.tail(150).to_dict(orient='records'), plot_data=plot_img, selected_market=selected_market, selected_fruit=selected_fruit, selected_variety=selected_variety, selected_grade=selected_grade, cards=cards)

        except Exception as e:
            logging.error(f"Dashboard error: {str(e)}")
            return render_template("dashboard.html", config=CONFIG, data=[], plot_data='[]', selected_market='', selected_fruit='', selected_variety='', selected_grade='', cards=[])
