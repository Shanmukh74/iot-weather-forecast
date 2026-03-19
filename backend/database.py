import pandas as pd
from datetime import datetime

DATA_FILE = "../data/weather_data.csv"
PREDICTION_FILE = "../data/predictions_log.csv"

# -------------------------
# Weather Data
# -------------------------
def get_all_data():
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = ["timestamp", "temperature", "humidity", "pressure", "rainfall"]
    return df

def get_latest_data():
    df = get_all_data()
    return df.iloc[-1]

# -------------------------
# Prediction Log
# -------------------------
def save_prediction(pred_time, target_time, predicted_temp):
    df = pd.DataFrame([{
        "prediction_time": pred_time,
        "target_time": target_time,
        "predicted_temperature": predicted_temp,
        "actual_temperature": None,
        "error": None
    }])

    try:
        existing = pd.read_csv(PREDICTION_FILE)
        df.to_csv(PREDICTION_FILE, mode='a', header=False, index=False)
    except FileNotFoundError:
        df.to_csv(PREDICTION_FILE, index=False)


def get_predictions():
    try:
        df = pd.read_csv(PREDICTION_FILE)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
