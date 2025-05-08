from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Path to your CSV file
CSV_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_time', 'Daily Crop_Fruit Prices.csv')

def load_data():
    df = pd.read_csv(CSV_FILE)

    # Strip and clean column names
    df.columns = [col.strip() for col in df.columns]
    df = df.drop(columns=['Sr. No.'], errors='ignore')
    # Drop rows with missing critical info
    df = df.dropna(subset=['Market Name', 'Fruit/Crop Variety', 'Date/Time', 'Price (per quintal)'])

    # Convert types
    df['Market Name'] = df['Market Name'].astype(str).str.strip()
    df['Fruit/Crop Variety'] = df['Fruit/Crop Variety'].astype(str).str.strip()
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    df['Price (per quintal)'] = pd.to_numeric(df['Price (per quintal)'], errors='coerce')

    # Drop rows again that may have failed conversion
    df = df.dropna(subset=['Date/Time', 'Price (per quintal)'])

    return df

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'svg'  # Use 'svg' for static HTML rendering

# ...

from datetime import timedelta
import pandas as pd

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    df = load_data()

    allowed_varieties = [
        'delicious-large', 'delicious-medium', 'delicious-small',
        'american-large', 'american-medium', 'american-small',
        'maharaji-large', 'maharaji-medium', 'maharaji-small',
        'golden-large', 'golden-medium', 'golden-small'
    ]
    df = df[df['Fruit/Crop Variety'].str.lower().isin(allowed_varieties)]
    df['Fruit/Crop Variety'] = df['Fruit/Crop Variety'].str.lower()

    # Time filter: Keep only records from the last 5 months
    five_months_ago = pd.Timestamp.now() - pd.DateOffset(months=5)
    df = df[df['Date/Time'] >= five_months_ago]

    markets = sorted(df['Market Name'].unique())
    varieties = sorted(df['Fruit/Crop Variety'].unique(), key=lambda x: allowed_varieties.index(x))

    selected_market = request.form.get('market', markets[0] if markets else "")
    selected_variety = request.form.get('variety', varieties[0] if varieties else "")

    filtered_df = df[(df['Market Name'] == selected_market) &
                     (df['Fruit/Crop Variety'] == selected_variety)].sort_values('Date/Time')

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date/Time'],
        y=filtered_df['Price (per quintal)'],
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=8),
        name='Price Trend'
    ))
    fig.update_layout(
        title=f'📈 Apple Price Trend - {selected_market} ({selected_variety})',
        xaxis_title='Date',
        yaxis_title='Price (₹/quintal)',
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    plot_html = pio.to_html(fig, full_html=False)

    return render_template('dashboard.html',
                           markets=markets,
                           varieties=varieties,
                           selected_market=selected_market,
                           selected_variety=selected_variety,
                           plot_html=plot_html,
                           table_data=filtered_df.to_html(classes='table table-striped table-hover', index=False))



if __name__ == '__main__':
    app.run(debug=True)
