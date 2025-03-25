# scripts/weekly_retrain.py

import os
import pandas as pd
from glob import glob
from src.model_training_test import train_and_evaluate_model  # adjust if needed

DATA_DIR = "data/real_time"

def load_latest_data():
    files = glob(os.path.join(DATA_DIR, "*.csv")) + glob(os.path.join(DATA_DIR, "*.xlsx"))
    latest = max(files, key=os.path.getctime)
    print(f"📂 Using file: {latest}")
    if latest.endswith(".csv"):
        return pd.read_csv(latest)
    else:
        return pd.read_excel(latest)

def retrain():
    df = load_latest_data()
    # Integrate your preprocessing from data_pipeline.py here
    print("🧠 Retraining model on new data...")
    train_and_evaluate_model(df)  # assumes this exists in your repo

if __name__ == "__main__":
    retrain()
