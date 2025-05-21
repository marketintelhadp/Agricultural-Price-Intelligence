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
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load LSTM model
model = load_model(Config.MODEL_PATH, custom_objects=Config.CUSTOM_OBJECTS)

# Load dataset
data = pd.read_csv(Config.DATA_PATH)
data = data[data['Mask'] == 1]

# Extract relevant price data
prices = data['Avg Price (per kg)'].values.reshape(-1, 1)

# Initialize and fit scaler
scaler = MinMaxScaler()
scaler.fit(prices)

# Get model input shape for time steps
input_shape = model.input_shape
time_steps = input_shape[1]  
logging.info(f"Model Input Shape: {input_shape}")

def create_trend_plot(past_prices, future_predictions):
    """Generate a trend plot of past prices and future predictions."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(past_prices)), past_prices, label='Past Prices', color='blue')
    plt.plot(range(len(past_prices), len(past_prices) + len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Trend Plot of Past Prices and Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (per kg)')
    plt.legend()
    plt.grid()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

def setup_routes(app):
    """Register routes for the Flask app."""

    @app.route('/')
    def home():
        try:
            # Scan models directory to build hierarchical structure
            base_model_dir = r'models'

            import os

            def build_hierarchy_model(path):
                """
                Build a nested dict hierarchy: market -> fruit -> variety -> grade -> model_file_path
                based on the directory structure and file names.
                Incorporate domain knowledge for special markets like Azadpur.
                """
                hierarchy = {}
                import os

                for market_entry in os.scandir(path):
                    if market_entry.is_dir():
                        market = market_entry.name
                        hierarchy[market] = {}

                        # Special handling for Azadpur market
                        if market.lower() == 'azadpur':
                            # Azadpur has varieties Makhmali, Misri with grades Fancy, Special, Super
                            for variety_entry in os.scandir(market_entry.path):
                                if variety_entry.is_dir():
                                    variety = variety_entry.name
                                    hierarchy[market][variety] = {}
                                    for grade_entry in os.scandir(variety_entry.path):
                                        if grade_entry.is_dir():
                                            grade = grade_entry.name
                                            # Look for .h5 files inside grade directory
                                            for file_entry in os.scandir(grade_entry.path):
                                                if file_entry.is_file() and file_entry.name.endswith('.h5'):
                                                    hierarchy[market][variety][grade] = file_entry.path
                                        else:
                                            # grade_entry is a file, treat as grade-less variety
                                            if variety not in hierarchy[market]:
                                                hierarchy[market][variety] = {}
                                            if grade_entry.name.endswith('.h5'):
                                                hierarchy[market][variety][''] = grade_entry.path
                        else:
                            # For other markets, assume structure: market -> fruit -> variety -> grade
                            for fruit_entry in os.scandir(market_entry.path):
                                if fruit_entry.is_dir():
                                    fruit = fruit_entry.name
                                    hierarchy[market][fruit] = {}
                                    for variety_entry in os.scandir(fruit_entry.path):
                                        if variety_entry.is_dir():
                                            variety = variety_entry.name
                                            hierarchy[market][fruit][variety] = {}
                                            for grade_entry in os.scandir(variety_entry.path):
                                                if grade_entry.is_dir():
                                                    grade = grade_entry.name
                                                    # Look for .h5 files inside grade directory
                                                    for file_entry in os.scandir(grade_entry.path):
                                                        if file_entry.is_file() and file_entry.name.endswith('.h5'):
                                                            hierarchy[market][fruit][variety][grade] = file_entry.path
                                                else:
                                                    # grade_entry is a file, treat as grade-less variety
                                                    if variety not in hierarchy[market][fruit]:
                                                        hierarchy[market][fruit][variety] = {}
                                                    if grade_entry.name.endswith('.h5'):
                                                        hierarchy[market][fruit][variety][''] = grade_entry.path
                                        else:
                                            # variety_entry is a file, treat as fruit with no variety/grade
                                            if fruit not in hierarchy[market]:
                                                hierarchy[market][fruit] = {}
                                            if variety_entry.name.endswith('.h5'):
                                                hierarchy[market][fruit][''] = variety_entry.path
                                else:
                                    # fruit_entry is a file, treat as market with no fruit/variety/grade
                                    if market not in hierarchy:
                                        hierarchy[market] = {}
                                    if fruit_entry.name.endswith('.h5'):
                                        hierarchy[market][''] = fruit_entry.path
                return hierarchy

            model_tree = build_hierarchy_model(base_model_dir)

            # Extract markets from model_tree keys
            markets = sorted(model_tree.keys())

            # Set default selections
            selected_market = markets[0] if markets else None
            fruits = sorted(model_tree.get(selected_market, {}).keys()) if selected_market and isinstance(model_tree.get(selected_market), dict) else []
            selected_fruit = fruits[0] if fruits else None
            varieties = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).keys()) if selected_market and selected_fruit and isinstance(model_tree.get(selected_market, {}).get(selected_fruit), dict) else []
            selected_variety = varieties[0] if varieties else None
            grades = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).get(selected_variety, {}).keys()) if selected_market and selected_fruit and selected_variety and isinstance(model_tree.get(selected_market, {}).get(selected_fruit, {}).get(selected_variety), dict) else []
            selected_grade = grades[0] if grades else None

            return render_template('predict.html',
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
            # Scan models directory to build hierarchical structure
            base_model_dir = r'models'
            base_data_dir = r'data/raw/processed'

            # Helper function to recursively scan directories and build nested dict
            import os

            def scan_dir(path):
                tree = {}
                for entry in os.scandir(path):
                    if entry.is_dir():
                        tree[entry.name] = scan_dir(entry.path)
                    else:
                        tree[entry.name] = entry.path
                return tree

            model_tree = scan_dir(base_model_dir)
            data_tree = scan_dir(base_data_dir)

            # Extract markets from model_tree keys
            markets = sorted(model_tree.keys())

            # Get selected values from form or defaults
            selected_market = request.form.get('market', markets[0] if markets else None)
            selected_fruit = request.form.get('fruit', None)
            selected_variety = request.form.get('variety', None)
            selected_grade = request.form.get('grade', None)
            num_predictions = int(request.form.get('num_predictions', 7))

            # Helper to get fruits for selected market
            fruits = sorted(model_tree.get(selected_market, {}).keys()) if selected_market else []
            if not selected_fruit and fruits:
                selected_fruit = fruits[0]

            # Helper to get varieties for selected fruit
            varieties = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).keys()) if selected_market and selected_fruit else []
            if not selected_variety and varieties:
                selected_variety = varieties[0]

            # Helper to get grades for selected variety
            grades = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).get(selected_variety, {}).keys()) if selected_market and selected_fruit and selected_variety else []
            if not selected_grade and grades:
                selected_grade = grades[0]

            # Helper function to find model file path given selections
            def find_model_path(base_path, market, fruit, variety, grade):
                market_path = os.path.join(base_path, market)
                if not os.path.exists(market_path):
                    return None
                for root, dirs, files in os.walk(market_path):
                    for file in files:
                        if file.endswith('.h5'):
                            filename_lower = file.lower()
                            variety_lower = variety.lower() if variety else ''
                            grade_lower = grade.lower() if grade else ''
                            fruit_lower = fruit.lower() if fruit else ''
                            if variety and grade:
                                if variety_lower in filename_lower and grade_lower in filename_lower:
                                    return os.path.join(root, file)
                            elif variety:
                                if variety_lower in filename_lower:
                                    return os.path.join(root, file)
                            else:
                                return os.path.join(root, file)
                return None

            # Helper function to find data file path given selections
            def find_data_path(base_path, market, fruit, variety, grade):
                market_path = os.path.join(base_path, market)
                if not os.path.exists(market_path):
                    return None
                possible_files = []
                if variety and grade:
                    possible_files.append(f"{variety}_{grade}_dataset.csv")
                    possible_files.append(f"{fruit}_{variety}_{grade}_dataset.csv")
                if variety:
                    possible_files.append(f"{variety}_dataset.csv")
                    possible_files.append(f"{fruit}_{variety}_dataset.csv")
                if fruit:
                    possible_files.append(f"{fruit}_dataset.csv")
                for root, dirs, files in os.walk(market_path):
                    for file in files:
                        if file in possible_files:
                            return os.path.join(root, file)
                return None

            model_path = find_model_path(base_model_dir, selected_market, selected_fruit, selected_variety, selected_grade)
            data_path = find_data_path(base_data_dir, selected_market, selected_fruit, selected_variety, selected_grade)

            if not data_path or not os.path.exists(data_path):
                return f"Dataset not found for selection: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            # Load dataset
            df = pd.read_csv(data_path)

            # Filter dataset: only rows where Mask == 1
            df = df[df['Mask'] == 1]

            # Filter dataset: only last 5 months data
            df['Date'] = pd.to_datetime(df['Date'])
            max_date = df['Date'].max()
            min_date = max_date - pd.DateOffset(months=5)
            df = df[(df['Date'] >= min_date) & (df['Date'] <= max_date)]

            prices = df['Avg Price (per kg)'].values.reshape(-1, 1)

            # Initialize and fit scaler
            scaler = MinMaxScaler()
            scaler.fit(prices)

            # Load model and perform forecasting
            if not model_path or not os.path.exists(model_path):
                return f"Model not found for selection: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            model = load_model(model_path, custom_objects=Config.CUSTOM_OBJECTS)

            input_shape = model.input_shape
            time_steps = input_shape[1]

            if len(prices) < time_steps:
                return jsonify({'error': 'Not enough data to make a prediction. Please provide more data.'}), 400

            last_sequence = prices[-time_steps:].reshape(-1, 1)
            scaled_sequence = scaler.transform(last_sequence)
            input_sequence = scaled_sequence.reshape(1, time_steps, 1)

            future_predictions = []
            for _ in range(num_predictions):
                prediction = model.predict(input_sequence)
                predicted_price = scaler.inverse_transform(prediction)
                future_predictions.append(float(predicted_price[0][0]))
                input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

            logging.info(f"Future predictions: {future_predictions}")

            # Get last 30 prices for trend plot
            past_prices = prices[-30:]
            trend_plot = create_trend_plot(past_prices, future_predictions)

            return render_template('predict.html',
                                   markets=markets,
                                   fruits=fruits,
                                   varieties=varieties,
                                   grades=grades,
                                   selected_market=selected_market,
                                   selected_fruit=selected_fruit,
                                   selected_variety=selected_variety,
                                   selected_grade=selected_grade,
                                   num_predictions=num_predictions,
                                   predicted_prices=zip([max_date + pd.Timedelta(days=i+1) for i in range(num_predictions)], future_predictions),
                                   trend_plot=trend_plot)

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

    @app.route('/dashboard', methods=['GET', 'POST'])
    def dashboard():
        try:
            # Scan models directory to build hierarchical structure
            base_model_dir = r'models'
            base_data_dir = r'data/raw/processed'

            # Helper function to recursively scan directories and build nested dict
            import os

            def scan_dir(path):
                tree = {}
                for entry in os.scandir(path):
                    if entry.is_dir():
                        tree[entry.name] = scan_dir(entry.path)
                    else:
                        tree[entry.name] = entry.path
                return tree

            model_tree = scan_dir(base_model_dir)
            data_tree = scan_dir(base_data_dir)

            # Extract markets from model_tree keys
            markets = sorted(model_tree.keys())

            # Get selected values from form or defaults
            selected_market = request.form.get('market', markets[0] if markets else None)
            selected_fruit = request.form.get('fruit', None)
            selected_variety = request.form.get('variety', None)
            selected_grade = request.form.get('grade', None)
            forecast_horizon = int(request.form.get('forecast_horizon', 7))

            # Helper to get fruits for selected market
            fruits = sorted(model_tree.get(selected_market, {}).keys()) if selected_market else []
            if not selected_fruit and fruits:
                selected_fruit = fruits[0]

            # Helper to get varieties for selected fruit
            varieties = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).keys()) if selected_market and selected_fruit else []
            if not selected_variety and varieties:
                selected_variety = varieties[0]

            # Helper to get grades for selected variety
            grades = sorted(model_tree.get(selected_market, {}).get(selected_fruit, {}).get(selected_variety, {}).keys()) if selected_market and selected_fruit and selected_variety else []
            if not selected_grade and grades:
                selected_grade = grades[0]

            # Helper function to find model file path given selections
            def find_model_path(base_path, market, fruit, variety, grade):
                import os
                market_path = os.path.join(base_path, market)
                if not os.path.exists(market_path):
                    return None
                # Traverse directories to find matching model file
                for root, dirs, files in os.walk(market_path):
                    for file in files:
                        if file.endswith('.h5'):
                            # Check if file matches pattern with variety and grade
                            # Use comments in filenames to match fruit, variety, grade
                            filename_lower = file.lower()
                            variety_lower = variety.lower() if variety else ''
                            grade_lower = grade.lower() if grade else ''
                            fruit_lower = fruit.lower() if fruit else ''
                            if variety and grade:
                                if variety_lower in filename_lower and grade_lower in filename_lower:
                                    return os.path.join(root, file)
                            elif variety:
                                if variety_lower in filename_lower:
                                    return os.path.join(root, file)
                            else:
                                return os.path.join(root, file)
                return None

            # Helper function to find data file path given selections
            def find_data_path(base_path, market, fruit, variety, grade):
                import os
                market_path = os.path.join(base_path, market)
                if not os.path.exists(market_path):
                    return None
                # Construct possible filenames based on user comments
                possible_files = []
                if variety and grade:
                    possible_files.append(f"{variety}_{grade}_dataset.csv")
                    possible_files.append(f"{fruit}_{variety}_{grade}_dataset.csv")
                if variety:
                    possible_files.append(f"{variety}_dataset.csv")
                    possible_files.append(f"{fruit}_{variety}_dataset.csv")
                if fruit:
                    possible_files.append(f"{fruit}_dataset.csv")
                # Search for files in market_path and subdirectories
                for root, dirs, files in os.walk(market_path):
                    for file in files:
                        if file in possible_files:
                            return os.path.join(root, file)
                return None

            model_path = find_model_path(base_model_dir, selected_market, selected_fruit, selected_variety, selected_grade)
            data_path = find_data_path(base_data_dir, selected_market, selected_fruit, selected_variety, selected_grade)

            if not data_path or not os.path.exists(data_path):
                return f"Dataset not found for selection: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            if not data_path or not os.path.exists(data_path):
                return f"Dataset not found for selection: {selected_market}, {selected_fruit}, {selected_variety}, {selected_grade}", 404

            # Load dataset
            df = pd.read_csv(data_path)

            # Filter dataset: only rows where Mask == 1
            df = df[df['Mask'] == 1]

            # Filter dataset: only last 5 months data
            df['Date'] = pd.to_datetime(df['Date'])
            max_date = df['Date'].max()
            min_date = max_date - pd.DateOffset(months=5)
            df = df[(df['Date'] >= min_date) & (df['Date'] <= max_date)]

            # Adjust to user request: table last 2 months, plot last 1 month
            table_min_date = max_date - pd.DateOffset(months=2)
            plot_min_date = max_date - pd.DateOffset(months=1)

            # Data for table: last 2 months
            df_display = df[df['Date'] >= table_min_date].drop(columns=['Mask'])

            # Data for plot: last 1 month
            df_plot = df[df['Date'] >= plot_min_date]

            # Generate price trend plot for last 1 month
            plt.figure(figsize=(10, 5))
            plt.plot(df_plot['Date'], df_plot['Avg Price (per kg)'], marker='o', label='Past Prices')
            plt.title(f'Price Trend for {selected_variety} {selected_grade} in {selected_market}')
            plt.xlabel('Date')
            plt.ylabel('Avg Price (per kg)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Generate HTML table for dataset
            table_data = df_display.to_html(classes='table table-striped', index=False)

            # Load model path
            model_path = None
            try:
                model_path = model_tree[selected_market][selected_fruit][selected_variety][selected_grade]
            except Exception:
                model_path = None

            forecasted_prices = []
            forecast_plot_html = None
            forecast_dates = []

            if model_path and os.path.exists(model_path):
                # Load model and perform forecasting for forecast_horizon days
                from keras.models import load_model
                from sklearn.preprocessing import MinMaxScaler
                import numpy as np
                import datetime

                model = load_model(model_path, custom_objects=Config.CUSTOM_OBJECTS)

                prices = df['Avg Price (per kg)'].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaler.fit(prices)

                input_shape = model.input_shape
                time_steps = input_shape[1]

                if len(prices) < time_steps:
                    forecasted_prices = ["Not enough data to forecast"]
                else:
                    last_sequence = prices[-time_steps:].reshape(-1, 1)
                    scaled_sequence = scaler.transform(last_sequence)
                    input_sequence = scaled_sequence.reshape(1, time_steps, 1)

                    future_predictions = []
                    for _ in range(forecast_horizon):
                        prediction = model.predict(input_sequence)
                        predicted_price = scaler.inverse_transform(prediction)
                        future_predictions.append(float(predicted_price[0][0]))
                        input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

                    forecasted_prices = future_predictions

                    # Generate future dates starting from max_date + 1 day
                    last_date = max_date
                    forecast_dates = [last_date + datetime.timedelta(days=i+1) for i in range(forecast_horizon)]

                    # Generate forecast plot with past 1 month and future forecasts
                    plt.figure(figsize=(10, 5))
                    plt.plot(df_plot['Date'], df_plot['Avg Price (per kg)'], label='Past Prices', color='blue')
                    plt.plot(forecast_dates, future_predictions, label='Forecasted Prices', color='orange')
                    plt.title(f'Forecast for {selected_variety} {selected_grade} in {selected_market}')
                    plt.xlabel('Date')
                    plt.ylabel('Price (per kg)')
                    plt.legend()
                    plt.grid()

                    img_forecast = io.BytesIO()
                    plt.savefig(img_forecast, format='png')
                    img_forecast.seek(0)
                    plt.close()
                    forecast_plot_html = base64.b64encode(img_forecast.getvalue()).decode()

            return render_template('dashboard.html',
                                   markets=markets,
                                   fruits=fruits,
                                   varieties=varieties,
                                   grades=grades,
                                   selected_market=selected_market,
                                   selected_fruit=selected_fruit,
                                   selected_variety=selected_variety,
                                   selected_grade=selected_grade,
                                   forecast_horizon=forecast_horizon,
                                   plot_html=f'<img src="data:image/png;base64,{plot_url}"/>',
                                   table_data=table_data,
                                   forecasted_prices=zip(forecast_dates, forecasted_prices) if forecast_dates else None,
                                   forecast_plot_html=f'<img src="data:image/png;base64,{forecast_plot_html}"/>' if forecast_plot_html else None)
        except Exception as e:
            logging.error(f"Error in dashboard route: {str(e)}")
            return "Error loading dashboard", 500
