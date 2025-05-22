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
from glob import glob
import os


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
def create_dashboard_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Price (₹/kg)"],
        mode="lines+markers", name="Historical Prices",
        line=dict(color="#ff7f0e")
    ))
    fig.update_layout(
        title="Recent Price Trends",
        xaxis_title="Date",
        yaxis_title="Price (₹/kg)",
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig.to_json()


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

    @app.route('/dashboard')
    def dashboard():
        try:
            selected_market = request.args.get('market', '')
            selected_fruit = request.args.get('fruit', '')
            selected_variety = request.args.get('variety', '')
            selected_grade = request.args.get('grade', '')

            base_dir = 'data/raw/processed'
            all_files = glob(os.path.join(base_dir, '**', '*_dataset.csv'), recursive=True)
            all_data = []

            for file_path in all_files:
                df = pd.read_csv(file_path)
                df = df[df['Mask'] == 1]
                df['Date'] = pd.to_datetime(df['Date'])

                parts = file_path.replace(base_dir, '').strip(os.sep).split(os.sep)
                if len(parts) >= 2:
                    market = parts[0]
                    file_parts = parts[-1].replace('_dataset.csv', '').split('_')

                    if len(file_parts) == 2:
                        variety, grade = file_parts
                    elif len(file_parts) == 3:
                        variety = file_parts[0] + ' ' + file_parts[1]
                        grade = file_parts[2]
                    else:
                        continue

                    known_fruits = ['Apple', 'Cherry']
                    fruit = next((f for f in known_fruits if f.lower() in file_path.lower()), 'Unknown')

                    df['Market'] = market
                    df['Fruit'] = fruit
                    df['Variety'] = variety
                    df['Grade'] = grade
                    df['Price'] = df['Avg Price (per kg)']
                    df.rename(columns={'Avg Price (per kg)': 'Price (₹/kg)'}, inplace=True)

                    all_data.append(df[['Date', 'Market', 'Fruit', 'Variety', 'Grade', 'Price (₹/kg)', 'Price']])

            if not all_data:
                raise ValueError("No valid datasets found")

            final_df = pd.concat(all_data)
            final_df.sort_values(by='Date', ascending=True, inplace=True)

            filtered_df = final_df.copy()
            if selected_market:
                filtered_df = filtered_df[filtered_df['Market'] == selected_market]
            if selected_fruit:
                filtered_df = filtered_df[filtered_df['Fruit'] == selected_fruit]
            if selected_variety:
                filtered_df = filtered_df[filtered_df['Variety'] == selected_variety]
            if selected_grade:
                filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]

            dropdown_data = {
                'markets': sorted(final_df['Market'].unique()),
                'fruits': sorted(final_df['Fruit'].unique()),
                'varieties': sorted(final_df['Variety'].unique()),
                'grades': sorted(final_df['Grade'].unique())
            }

            return render_template("dashboard.html",
                data=filtered_df.tail(100).to_dict(orient='records'),
                plot_data=create_dashboard_plot(filtered_df.tail(100)),
                dropdown_options=dropdown_data,
                selected_market=selected_market,
                selected_fruit=selected_fruit,
                selected_variety=selected_variety,
                selected_grade=selected_grade
            )

        except Exception as e:
            logging.error(f"Error in dashboard: {str(e)}")
            return render_template("dashboard.html",
                                data=[],
                                plot_data='[]',
                                dropdown_options={},
                                selected_market='',
                                selected_fruit='',
                                selected_variety='',
                                selected_grade='')
