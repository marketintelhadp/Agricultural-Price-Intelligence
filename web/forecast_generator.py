from keras.models import load_model
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from web.config import CONFIG
from web.routes import forecast_sequence, sale_periods
import traceback
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os


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
    ('Narwal', 'American', 'Unknown'): {'start': '09-01', 'end': '12-30', 'years': list(range(2011, 2026))},
    ('Narwal', 'Cherry', 'Large'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Medium'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Small'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Condition', 'Unknown'): {'start': '07-01', 'end': '08-31', 'years': list(range(2019, 2026))},
    ('Narwal', 'Delicious', 'Unknown'): {'start': '01-01', 'end': '12-31', 'years': list(range(2011, 2026))},
    ('Narwal', 'Hazratbali', 'Unknown'): {'start': '07-01', 'end': '08-30', 'years': list(range(2011, 2026))},
    ('Narwal', 'Razakwadi', 'Unknown'): {'start': '08-01', 'end': '08-30', 'years': list(range(2013, 2026))},
    ('Parimpore', 'Cherry', 'Large'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Medium'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Small'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Shopian', 'American', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'American', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Cherry', 'Large'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Medium'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Small'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Delicious', 'A'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Delicious', 'B'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu delicious', 'A'): {'start': '09-01', 'end': '12-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu delicious', 'B'): {'start': '09-01', 'end': '12-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Sopore', 'American', 'A'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'American', 'B'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'A'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'B'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'A'): {'start': '11-01', 'end': '12-30', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'B'): {'start': '11-01', 'end': '12-30', 'years': list(range(2015, 2026))}
}

def get_forecast_start_dates(market, variety, grade, forecast_days):
    key = (market, variety, grade)
    if key not in sale_periods:
        raise ValueError(f"No sale period defined for {market}, {variety}, {grade}.")

    sale_info = sale_periods[key]
    current_year = datetime.today().year
    start_date = datetime.strptime(f"{current_year}-{sale_info['start']}", "%Y-%m-%d")
    return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]

def generate_forecast(model_path, dataset_path, forecast_dates):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    model = load_model(model_path)
    df = pd.read_csv(dataset_path)
    prices = df['price'].values
    window_size = 30

    if len(prices) < window_size:
        raise ValueError("Insufficient historical data for forecasting.")

    last_window = prices[-window_size:].reshape(1, window_size, 1)
    predictions = []

    for _ in forecast_dates:
        pred = model.predict(last_window, verbose=0)[0][0]
        predictions.append(pred)
        last_window = np.append(last_window[0][1:], [[pred]], axis=0).reshape(1, window_size, 1)

    forecast_result = list(zip(forecast_dates, predictions))
    return forecast_result

# Example function to integrate with route

def forecast_prices(market, fruit, variety, grade, forecast_option, config):
    forecast_days = 7 if forecast_option == 'week' else 15

    if market not in config or fruit not in config[market] or \
       variety not in config[market][fruit] or grade not in config[market][fruit][variety]:
        raise ValueError("Invalid market/fruit/variety/grade combination.")

    model_info = config[market][fruit][variety][grade]
    model_path = model_info['model']
    dataset_path = model_info['dataset']

    forecast_dates = get_forecast_start_dates(market, variety, grade, forecast_days)
    predictions = generate_forecast(model_path, dataset_path, forecast_dates)

    return predictions, forecast_dates[0]  # Also return the start date for UI message


def precompute_forecasts():
    current_date = datetime.now().date()
    today = pd.to_datetime(current_date)
    print("\n📊 Starting forecast precomputation...")

    for market in CONFIG:
        for fruit in CONFIG[market]:
            if fruit != "Cherry":
                continue
            for variety in CONFIG[market][fruit]:
                for grade in CONFIG[market][fruit][variety]:
                    key = (market, variety, grade)
                    try:
                        print(f"\n🔍 Processing: {key}")

                        # Load config and model
                        entry = CONFIG[market][fruit][variety][grade]
                        model_path = entry['model']
                        dataset_path = entry['dataset']
                        print(f"  → Model: {model_path}")
                        print(f"  → Dataset: {dataset_path}")

                        model = load_model(model_path, compile=False)
                        df = pd.read_csv(dataset_path)

                        df = df[df['Mask'] == 1]
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df[df['Date'] <= today]
                        df.sort_values(by='Date', inplace=True)

                        prices = df['Avg Price (per kg)'].values
                        time_steps = model.input_shape[1]

                        if len(prices) < time_steps:
                            print("  ⚠ Skipping due to insufficient data.")
                            continue

                        last_seq = prices[-time_steps:].reshape(-1, 1)
                        scaler = MinMaxScaler().fit(last_seq)
                        input_seq = scaler.transform(last_seq).reshape(1, time_steps, 1)

                        sale_info = sale_periods.get(key)
                        if not sale_info:
                            print("  ⚠ Skipping due to missing sale_periods entry.")
                            continue

                        # Build forecast_start date from current year
                        forecast_start = pd.to_datetime(f"{today.year}-{sale_info['start']}")

                        # If current date is before forecast_start, shift start to forecast_start
                        start_date = max(today, forecast_start)

                        forecast_end = pd.to_datetime(f"{today.year}-{sale_info['end']}")
                        if start_date > forecast_end:
                            print("  ⚠ Sale period has already ended.")
                            continue

                        total_days = (forecast_end - start_date).days + 1

                        if total_days <= 0:
                            print("  ⚠ Sale period has already ended.")
                            continue

                        forecasted_prices = forecast_sequence(model, input_seq, total_days, scaler)
                        forecast_dates = pd.date_range(start=start_date, periods=total_days)

                        out_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecasted_prices
                        })

                        safe_market = market.replace(' ', '_').strip()
                        safe_variety = variety.replace(' ', '_').strip()
                        safe_grade = grade.replace(' ', '_').strip()

                        out_path = f"data/forecasts/{safe_market}_{safe_variety}_{safe_grade}_forecast.csv"
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        out_df.to_csv(out_path, index=False)

                        print(f"  ✅ Forecast saved: {out_path}")

                    except Exception as e:
                        print(f"  ❌ Failed for {key}: {e}")
                        traceback.print_exc()

if __name__ == '__main__':
    precompute_forecasts()
    print("\n✅ All precomputations done. Check `data/forecasts/` for results.")
