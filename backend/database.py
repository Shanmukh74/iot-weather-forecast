import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/weather_data.csv")
PRED_PATH = os.path.join(BASE_DIR, "../data/predictions_log.csv")


# ----------------------------
# GET ALL WEATHER DATA
# ----------------------------
def get_all_data():
    try:
        print("Reading CSV from:", DATA_PATH)

        if not os.path.exists(DATA_PATH):
            print("CSV NOT FOUND")
            return None

        df = pd.read_csv(DATA_PATH)

        if df.empty:
            print("CSV EMPTY")
            return None

        # FIX column names (important)
        df.columns = df.columns.str.strip()

        # Convert timestamp safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

        # Drop invalid rows
        df = df.dropna()

        print("ROWS LOADED:", len(df))

        return df

    except Exception as e:
        print("ERROR reading CSV:", e)
        return None


# ----------------------------
# GET LATEST ROW
# ----------------------------
def get_latest_data():
    df = get_all_data()

    if df is None or df.empty:
        return None

    return df.iloc[-1]   # ✅ return actual row


# ----------------------------
# SAVE PREDICTION
# ----------------------------
def save_prediction(pred_time, target_time, prediction):

    data = {
        "prediction_time": pred_time,
        "target_time": target_time,
        "predicted_temperature": prediction,
        "actual_temperature": None,
        "error": None
    }

    df = pd.DataFrame([data])

    if not os.path.exists(PRED_PATH):
        df.to_csv(PRED_PATH, index=False)
    else:
        df.to_csv(PRED_PATH, mode='a', header=False, index=False)


# ----------------------------
# GET PREDICTIONS
# ----------------------------
def get_predictions():
    try:
        if not os.path.exists(PRED_PATH):
            return pd.DataFrame()

        df = pd.read_csv(PRED_PATH)

        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        print("ERROR reading predictions:", e)
        return pd.DataFrame()