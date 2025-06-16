import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import lru_cache
from flask import request, jsonify, render_template, flash
import numpy as np
import pandas as pd
import logging
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError
from web.config import CONFIG
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
from sqlalchemy import create_engine
from flask import Blueprint, render_template, request, flash
from sqlalchemy import create_engine, text
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mydash_bp = Blueprint('mydash', __name__, template_folder='templates')

# PostgreSQL connection
DATABASE_URL = "postgresql://marketdata_m0dt_user:jSdEzjqgKTdeqmjQIwr8UIRBa3qglzxD@dpg-d0inpmqdbo4c738msb60-a.oregon-postgres.render.com/marketdata_m0dt"
engine = create_engine(DATABASE_URL)


sale_periods = {
    ('Azadpur', 'Makhmali', 'Fancy'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Makhmali', 'Special'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Makhmali', 'Super'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Misri', 'Fancy'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Azadpur', 'Misri', 'Special'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Azadpur', 'Misri', 'Super'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Ganderbal', 'Cherry', 'Large'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Ganderbal', 'Cherry', 'Medium'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Ganderbal', 'Cherry', 'Small'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Narwal', 'Cherry', 'Large'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Medium'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Small'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'American', '_'): {'start': '09-01', 'end': '12-31', 'years': list(range(2011, 2026))},
    ('Narwal', 'Condition', '_'): {'start': '07-01', 'end': '08-31', 'years': list(range(2019, 2026))},
    ('Narwal', 'Delicious', '_'): {'start': '08-01', 'end': '02-28', 'years': list(range(2011, 2026))},
    ('Narwal', 'Hazratbali', '_'): {'start': '07-01', 'end': '08-31', 'years': list(range(2011, 2026))},
    ('Narwal', 'Razakwadi', '_'): {'start': '08-01', 'end': '08-31', 'years': list(range(2013, 2026))},
    ('Parimpore', 'Cherry', 'Large'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Medium'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Small'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Shopian', 'American', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'American', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Cherry', 'Large'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Medium'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Small'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'American', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'American', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Delicious', 'A'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Delicious', 'B'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu Delicious', 'A'): {'start': '09-01', 'end': '11-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu Delicious', 'B'): {'start': '09-01', 'end': '11-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Pulwama-Pachhar', 'American', 'A'): {'start': '09-15', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'American', 'A'): {'start': '09-15', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'Delicious', 'A'): {'start': '09-15', 'end': '12-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'Kullu Delicious', 'A'): {'start': '09-01', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'American', 'A'): {'start': '09-15', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'Delicious', 'A'): {'start': '09-15', 'end': '12-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'Kullu Delicious', 'A'): {'start': '09-01', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'American', 'B'): {'start': '09-15', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'Delicious', 'B'): {'start': '09-15', 'end': '12-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Pachhar', 'Kullu Delicious', 'B'): {'start': '09-01', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'American', 'B'): {'start': '09-15', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'Delicious', 'B'): {'start': '09-15', 'end': '12-15', 'years': list(range(2018, 2026))},
    ('Pulwama-Prichoo', 'Kullu Delicious', 'B'): {'start': '09-01', 'end': '11-15', 'years': list(range(2018, 2026))},
    ('Sopore', 'American', 'A'): {'start': '08-01', 'end': '02-28', 'years': list(range(2015, 2026))},
    ('Sopore', 'American', 'B'): {'start': '08-01', 'end': '02-28', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'A'): {'start': '08-01', 'end': '02-28', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'B'): {'start': '08-01', 'end': '02-28', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'A'): {'start': '11-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'B'): {'start': '11-01', 'end': '12-31', 'years': list(range(2015, 2026))}
}

def create_forecast_plot(forecast_dates, future_predictions):
    trace = go.Scatter(x=forecast_dates, y=future_predictions, mode='lines+markers', name='Forecast')
    layout = go.Layout(
        title='Forecasted Prices',
        xaxis_title='Date',
        yaxis_title='Price (per kg)',
        template='none',  # SAFE TEMPLATE
        margin=dict(l=30, r=30, t=50, b=30)
    )
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)


def align_forecast_dates_to_previous_year(df, forecast_days, target_year):
    df = df.copy()
    df['MonthDay'] = df['Date'].dt.strftime('%m-%d')
    unique_md = sorted(df['MonthDay'].unique())
    if len(unique_md) < forecast_days:
        raise ValueError("Not enough date variety in past data for forecast window")
    return [f"{target_year}-{md}" for md in unique_md[:forecast_days]]


def create_marketdata_plot(df):
    import plotly.graph_objs as go
    import plotly.io as pio

    # Step 1: Format date and sort
    df['Submission Date'] = pd.to_datetime(df['Submission Date'])
    df.sort_values('Submission Date', inplace=True)

    # Color maps
    demand_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    supply_symbols = {'High': 'star', 'Medium': 'diamond', 'Low': 'circle'}

    # Step 2: Rich hover text
    df['HoverText'] = (
        "<b>Market:</b> " + df['Market'].astype(str) + "<br>" +
        "<b>Fruit:</b> " + df['Fruit'].astype(str) + "<br>" +
        "<b>Variety:</b> " + df['Variety'].astype(str) + "<br>" +
        "<b>Grade:</b> " + df['Grade'].astype(str) + "<br>" +
        "<b>Min Price:</b> ₹" + df['Min Price'].astype(str) + "<br>" +
        "<b>Max Price:</b> ₹" + df['Max Price'].astype(str) + "<br>" +
        "<b>Modal Price:</b> ₹" + df['Price (₹/kg)'].astype(str) + "<br>" +
        "<b>Arrival Qty:</b> " + df['Arrival Qty'].astype(str) + " MT<br>" +
        "<b>Transaction Volume:</b> " + df['Transaction Volume'].astype(str) + "<br>" +
        "<b>Stock:</b> " + df['Stock'].astype(str) + "<br>" +
        "<b>Demand:</b> " + df['Demand'].astype(str) + "<br>" +
        "<b>Supply:</b> " + df['Supply'].astype(str) + "<br>" +
        "<b>Weather:</b> " + df['Weather'].astype(str)
    )

    # Step 3: Create grouped traces, skip Apple
    traces = []
    grouped = df.groupby(['Market', 'Fruit'])

    for (market, fruit), group in grouped:
        if fruit.strip().lower() == 'apple':
            continue  # Skip Apple in plot

        trace = go.Scatter(
            x=group['Submission Date'],
            y=group['Price (₹/kg)'],
            mode='markers+lines',
            name=f"{market} - {fruit}",
            text=group['HoverText'],
            hoverinfo='text',
            marker=dict(
                size=10,
                color=[demand_colors.get(x, 'gray') for x in group['Demand']],
                symbol=[supply_symbols.get(x, 'circle') for x in group['Supply']]
            ),
            connectgaps=False
        )
        traces.append(trace)

    # Step 4: Layout
    layout = go.Layout(
        title='🧺 Market Intelligence: Modal Price Trends with Demand-Supply Cues',
        xaxis_title='Submission Date',
        yaxis_title='Modal Price (₹/kg)',
        template='plotly_white',
        margin=dict(l=40, r=30, t=60, b=40),
        hovermode='closest',
        legend_title_text='Market - Fruit'
    )

    fig = go.Figure(data=traces, layout=layout)
    return pio.to_html(fig, full_html=False)


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
    

from tensorflow.keras.models import load_model

@lru_cache(maxsize=10)
def load_model_cached(model_path):
    return load_model(model_path, compile=False)

def forecast_sequence(model, input_seq, days, scaler):
    """Efficient batch prediction loop."""
    forecast = []
    seq = input_seq.copy()
    for _ in range(days):
        pred = model.predict(seq, verbose=0)
        forecast.append(pred[0, 0])
        seq = np.concatenate([seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()



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

            sale_periods_json = { "|".join(k): v for k, v in sale_periods.items() }
            return render_template('predict.html',
                               config=CONFIG,
                               sale_periods=sale_periods_json,  # 🛠 Add this line
                               markets=markets,
                               fruits=fruits,
                               varieties=varieties,
                               grades=grades,
                               selected_market=selected_market,
                               selected_fruit=selected_fruit,
                               selected_variety=selected_variety,
                               selected_grade=selected_grade)
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
            selected_submarket = request.form.get('submarket')
            forecast_option = request.form.get('forecast_option')  # 'week' or 'fortnight'

            # Map Pulwama submarkets to sale_periods keys
            if selected_market == "Pulwama":
                if not selected_submarket:
                    return jsonify({'error': 'Submarket selection is required for Pulwama.'}), 400
                sale_market = f"Pulwama-{selected_submarket}"
            elif selected_market == "Pachhar Pulwama":
                sale_market = "Pulwama-Pachhar"
            elif selected_market == "Prichoo Pulwama":
                sale_market = "Pulwama-Prichoo"
            else:
                sale_market = selected_market

            # ✅ Use today's date as start_date
            start_date = pd.to_datetime(datetime.today().date())

            if not forecast_option:
                return jsonify({'error': 'Forecast option is required.'}), 400

            sale_key = (sale_market, selected_variety, selected_grade)
            sale_info = sale_periods.get(sale_key)
            if not sale_info:
                return jsonify({'error': f'No sale period defined for {sale_market}, {selected_variety}, {selected_grade}.'}), 400

            current_year = datetime.now().year
            if current_year not in sale_info['years']:
                return jsonify({'error': f'No sale period defined for {selected_market}, {selected_variety}, {selected_grade} in {current_year}.'}), 400

            sale_start_date = pd.to_datetime(f"{current_year}-{sale_info['start']}")
            sale_end_date = pd.to_datetime(f"{current_year}-{sale_info['end']}")

            # Adjust start_date to be within sale period
            start_date = max(start_date, sale_start_date)
            end_date = min(datetime.now(), sale_end_date)

            if start_date > end_date:
                return jsonify({'error': f'No forecast available for today. Valid range: {sale_start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'}), 400

            # ✅ Load precomputed forecast CSV
            forecast_file = f"data/forecasts/{selected_market}_{selected_variety}_{selected_grade}_forecast.csv"
            if not os.path.exists(forecast_file):
                return jsonify({'error': 'Forecast data not yet generated. Please try later.'}), 500

            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

            # ✅ Filter forecast from selected start_date
            forecast_df = forecast_df[forecast_df['Date'] >= start_date]
            forecast_length = 7 if forecast_option == 'week' else 14
            filtered = forecast_df.head(forecast_length)

            if filtered.empty:
                return jsonify({'error': f'No forecasts available starting from {start_date.strftime("%Y-%m-%d")}.'}), 400

            forecast_dates = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()
            filtered_prices = filtered['Forecast'].tolist()
            forecast_plot = create_forecast_plot(forecast_dates, filtered_prices)
            predicted_prices = list(zip(forecast_dates, filtered_prices))

            # Prepare dropdown reload
            fruits, varieties, grades = get_config_options(selected_market, selected_fruit, selected_variety)
            sale_periods_json = {"|".join(k): v for k, v in sale_periods.items()}

            return render_template('predict.html',
                config=CONFIG,
                sale_periods=sale_periods_json,
                markets=sorted(CONFIG.keys()),
                fruits=fruits,
                varieties=varieties,
                grades=grades,
                selected_market=selected_market,
                selected_fruit=selected_fruit,
                selected_variety=selected_variety,
                selected_grade=selected_grade,
                predicted_prices=predicted_prices,
                trend_plot=forecast_plot,
                start_date=start_date.strftime('%Y-%m-%d'))

        except Exception as e:
            logging.exception("Prediction failed with traceback:")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


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


    @mydash_bp.route('/mydash')
    def mydash():
        try:
            sql = text("""
                SELECT * FROM market_data
                ORDER BY submission_date DESC
                LIMIT 150
            """)
            df = pd.read_sql(sql, engine)

            if df.empty:
                flash("No data found in the database.", "warning")
                return render_template("mydash.html", data=[], plot_data='')

            df['Date'] = pd.to_datetime(df['submission_date'])
            df.rename(columns={
                'modal_price': 'Price (₹/kg)',
                'min_price': 'Min Price',
                'max_price': 'Max Price',
                'arrival_qty': 'Arrival Qty',
                'transaction_volume': 'Transaction Volume',
                'stock': 'Stock',
                'market': 'Market',
                'fruit': 'Fruit',
                'variety': 'Variety',
                'grade': 'Grade',
                'demand': 'Demand',
                'supply': 'Supply',
                'weather': 'Weather',
                'submission_date': 'Submission Date'
            }, inplace=True)

            df['Price'] = df['Price (₹/kg)']

            data = df.to_dict(orient='records')
            plot_img = create_marketdata_plot(df)

            return render_template("mydash.html", data=data, plot_data=plot_img)

        except Exception as e:
            logging.error(f"mydash error: {e}")
            flash("An error occurred while loading dashboard.", "danger")
            return render_template("mydash.html", data=[], plot_data='')