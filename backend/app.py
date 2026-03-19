from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta
import os

from database import get_latest_data, save_prediction, get_predictions, get_all_data

app = Flask(__name__)
CORS(app)

# ------------------------------------------------
# FIX PATH FOR MODEL (IMPORTANT FOR RENDER)
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/temperature_forecast_model.pkl")

model = joblib.load(MODEL_PATH)


# ------------------------------------------------
# BUILD INPUT FEATURES (Lag-based)
# ------------------------------------------------
def build_input_features():
    df = get_all_data()

    if df is None or len(df) < 13:
        return None

    df = df.reset_index(drop=True)

    return pd.DataFrame([{
        "temperature": float(df.iloc[-1]["temperature"]),
        "humidity": float(df.iloc[-1]["humidity"]),
        "pressure": float(df.iloc[-1]["pressure"]),
        "rainfall": float(df.iloc[-1]["rainfall"]),
        "temp_lag_1": float(df.iloc[-2]["temperature"]),
        "temp_lag_2": float(df.iloc[-3]["temperature"]),
        "temp_lag_6": float(df.iloc[-7]["temperature"]),
        "temp_lag_12": float(df.iloc[-13]["temperature"])
    }])


# ------------------------------------------------
# CURRENT WEATHER API
# ------------------------------------------------
@app.route("/current")
def current_weather():
    latest = get_latest_data()

    if latest is None or len(latest) == 0:
        return jsonify({"message": "No data available"})

    latest = latest.iloc[0]

    return jsonify({
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "pressure": float(latest["pressure"]),
        "rainfall": float(latest["rainfall"])
    })


# ------------------------------------------------
# PREDICTION API
# ------------------------------------------------
@app.route("/predict")
def predict_weather():

    input_data = build_input_features()

    if input_data is None:
        return jsonify({"message": "Not enough data"})

    prediction = float(model.predict(input_data)[0])

    prediction_time = datetime.now()
    target_time = prediction_time + timedelta(minutes=10)

    save_prediction(prediction_time, target_time, prediction)

    return jsonify({
        "forecast_temperature_10min": prediction,
        "target_time": target_time.isoformat()
    })


# ------------------------------------------------
# AUTO EVALUATION
# ------------------------------------------------
def auto_evaluate():
    predictions = get_predictions()
    weather = get_all_data()

    if predictions is None or predictions.empty:
        return

    if weather is None or weather.empty:
        return

    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    updated = False

    for index, row in predictions.iterrows():

        if pd.isna(row["actual_temperature"]):

            target_time = pd.to_datetime(row["target_time"])

            if datetime.now() >= target_time:

                closest = weather.iloc[
                    (weather["timestamp"] - target_time).abs().argsort()[:1]
                ]

                actual_temp = float(closest["temperature"].values[0])
                predicted_temp = float(row["predicted_temperature"])

                error = abs(actual_temp - predicted_temp)

                predictions.at[index, "actual_temperature"] = actual_temp
                predictions.at[index, "error"] = error

                updated = True

    if updated:
        DATA_PATH = os.path.join(BASE_DIR, "../data/predictions_log.csv")
        predictions.to_csv(DATA_PATH, index=False)


# ------------------------------------------------
# ACCURACY API
# ------------------------------------------------
@app.route("/accuracy")
def accuracy():
    auto_evaluate()

    predictions = get_predictions()

    if predictions is None or predictions.empty:
        return jsonify({"message": "No predictions yet"})

    valid_errors = predictions["error"].dropna()

    if valid_errors.empty:
        return jsonify({"message": "No evaluated predictions yet"})

    mae = float(valid_errors.mean())

    return jsonify({
        "Average_MAE": mae,
        "Total_Evaluated": int(valid_errors.count())
    })


# ------------------------------------------------
# FORECAST HISTORY
# ------------------------------------------------
@app.route("/history")
def history():
    auto_evaluate()

    predictions = get_predictions()

    if predictions is None or predictions.empty:
        return jsonify([])

    predictions = predictions.dropna()

    if predictions.empty:
        return jsonify([])

    return jsonify(
        predictions.tail(10).to_dict(orient="records")
    )


# ------------------------------------------------
# LIVE TEMPERATURE HISTORY
# ------------------------------------------------
@app.route("/live-history")
def live_history():
    weather = get_all_data()

    if weather is None or weather.empty:
        return jsonify([])

    weather = weather.tail(30)

    return jsonify(
        weather[["timestamp", "temperature"]].to_dict(orient="records")
    )


# ------------------------------------------------
# ROOT CHECK (IMPORTANT FOR RENDER)
# ------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "IoT Weather Backend Running"})


# ------------------------------------------------
# RUN APP (RENDER READY)
# ------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))