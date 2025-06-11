from keras.models import load_model
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from web.config import CONFIG
from web.routes import forecast_sequence, sale_periods
import traceback

def precompute_forecasts():
    today = pd.to_datetime(datetime.now().date())
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

                        end_date = pd.to_datetime(f"{today.year}-{sale_info['end']}")
                        total_days = (end_date - today).days + 1

                        if total_days <= 0:
                            print("  ⚠ Sale period has already ended.")
                            continue

                        forecasted_prices = forecast_sequence(model, input_seq, total_days, scaler)
                        forecast_dates = pd.date_range(start=today, periods=total_days)

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
