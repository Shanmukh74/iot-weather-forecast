from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import requests

app = Flask(__name__)
CORS(app)

# ------------------------------------------------
# IST TIMEZONE
# ------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

# ------------------------------------------------
# THINGSPEAK CONFIG
# ------------------------------------------------
CHANNEL_ID = "3308123"

# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/temperature_forecast_model.pkl")
PRED_PATH = os.path.join(BASE_DIR, "../data/predictions_log.csv")

model = joblib.load(MODEL_PATH)

# ------------------------------------------------
# READ DATA FROM THINGSPEAK
# ------------------------------------------------
def read_weather_data():
    try:
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results=30"
        response = requests.get(url)

        if response.status_code != 200:
            print("API Error:", response.text)
            return None

        data = response.json()

        if not isinstance(data, dict) or "feeds" not in data:
            return None

        df = pd.DataFrame(data["feeds"])

        df["temperature"] = pd.to_numeric(df["field1"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["field2"], errors="coerce")
        df["pressure"] = pd.to_numeric(df["field3"], errors="coerce")
        df["rainfall"] = pd.to_numeric(df["field4"], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(IST)

        df = df.dropna()

        return df

    except Exception as e:
        print("ERROR:", e)
        return None


# ------------------------------------------------
# SAVE PREDICTION TO CSV
# ------------------------------------------------
def save_prediction(pred_time, target_time, prediction):

    data = {
        "prediction_time": pred_time.isoformat(),
        "target_time": target_time.isoformat(),
        "predicted_temperature": prediction,
        "actual_temperature": None,
        "error": None
    }

    df = pd.DataFrame([data])

    if not os.path.exists(PRED_PATH):
        df.to_csv(PRED_PATH, index=False)
    else:
        df.to_csv(PRED_PATH, mode='a', header=False, index=False)


# ------------------------------------------------
# LOAD PREDICTIONS
# ------------------------------------------------
def get_predictions():
    if not os.path.exists(PRED_PATH):
        return pd.DataFrame()

    df = pd.read_csv(PRED_PATH)

    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce", utc=True).dt.tz_convert(IST)
    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce", utc=True).dt.tz_convert(IST)

    return df


# ------------------------------------------------
# CURRENT WEATHER
# ------------------------------------------------
@app.route("/current")
def current_weather():
    df = read_weather_data()

    if df is None or df.empty:
        return jsonify({
            "temperature": None,
            "humidity": None,
            "pressure": None,
            "rainfall": None
        })

    latest = df.iloc[-1]

    return jsonify({
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "pressure": float(latest["pressure"]),
        "rainfall": float(latest["rainfall"])
    })


# ------------------------------------------------
# BUILD FEATURES
# ------------------------------------------------
def build_input_features():
    df = read_weather_data()

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
# PREDICT
# ------------------------------------------------
@app.route("/predict")
def predict_weather():
    input_data = build_input_features()

    if input_data is None:
        return jsonify({"message": "Not enough data"})

    prediction = float(model.predict(input_data)[0])

    prediction_time = datetime.now(IST)
    target_time = prediction_time + timedelta(minutes=10)

    # Save to CSV
    save_prediction(prediction_time, target_time, prediction)

    return jsonify({
        "forecast_temperature_10min": prediction,
        "target_time": target_time.isoformat()
    })


# ------------------------------------------------
# AUTO EVALUATE
# ------------------------------------------------
def auto_evaluate():
    predictions = get_predictions()
    weather = read_weather_data()

    if predictions.empty or weather is None:
        return

    for i, row in predictions.iterrows():

        if pd.isna(row["actual_temperature"]):

            if datetime.now(IST) >= row["target_time"]:

                closest = weather.iloc[
                    (weather["timestamp"] - row["target_time"]).abs().argsort()[:1]
                ]

                actual = float(closest["temperature"].values[0])
                predicted = float(row["predicted_temperature"])

                predictions.at[i, "actual_temperature"] = actual
                predictions.at[i, "error"] = abs(actual - predicted)

    predictions.to_csv(PRED_PATH, index=False)


# ------------------------------------------------
# ACCURACY
# ------------------------------------------------
@app.route("/accuracy")
def accuracy():
    auto_evaluate()

    df = get_predictions()

    valid = df["error"].dropna()

    if valid.empty:
        return jsonify({"message": "No evaluated predictions yet"})

    return jsonify({
        "Average_MAE": float(valid.mean()),
        "Total_Evaluated": int(valid.count())
    })


# ------------------------------------------------
# HISTORY
# ------------------------------------------------
@app.route("/history")
def history():
    auto_evaluate()

    df = get_predictions().dropna()

    return jsonify(df.tail(10).to_dict(orient="records"))


# ------------------------------------------------
# LIVE HISTORY
# ------------------------------------------------
@app.route("/live-history")
def live_history():
    df = read_weather_data()

    if df is None:
        return jsonify([])

    return jsonify(
        df.tail(30)[["timestamp", "temperature"]].to_dict(orient="records")
    )


# ------------------------------------------------
# ROOT
# ------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "IoT Weather Backend Running (FINAL CSV MODE)"})


# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)