from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta
from database import get_latest_data, save_prediction, get_predictions, get_all_data

app = Flask(__name__)
CORS(app)

model = joblib.load("../model/temperature_forecast_model.pkl")


# ------------------------------------------------
# Build Model Input (Lag Features)
# ------------------------------------------------
def build_input_features():
    df = get_all_data()

    if len(df) < 13:
        return None

    latest = df.iloc[-1]

    return pd.DataFrame([{
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "pressure": float(latest["pressure"]),
        "rainfall": float(latest["rainfall"]),
        "temp_lag_1": float(df.iloc[-2]["temperature"]),
        "temp_lag_2": float(df.iloc[-3]["temperature"]),
        "temp_lag_6": float(df.iloc[-7]["temperature"]),
        "temp_lag_12": float(df.iloc[-13]["temperature"])
    }])


# ------------------------------------------------
# Current Weather API
# ------------------------------------------------
@app.route("/current")
def current_weather():
    latest = get_latest_data().to_dict()

    return jsonify({
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "pressure": float(latest["pressure"]),
        "rainfall": float(latest["rainfall"])
    })


# ------------------------------------------------
# Predict API
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
        "target_time": str(target_time)
    })


# ------------------------------------------------
# Auto Evaluation
# ------------------------------------------------
def auto_evaluate():
    predictions = get_predictions()
    weather = get_all_data()

    if predictions.empty:
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
                error = abs(actual_temp - float(row["predicted_temperature"]))

                predictions.at[index, "actual_temperature"] = actual_temp
                predictions.at[index, "error"] = error

                updated = True

    if updated:
        predictions.to_csv("../data/predictions_log.csv", index=False)


# ------------------------------------------------
# Accuracy API
# ------------------------------------------------
@app.route("/accuracy")
def accuracy():
    auto_evaluate()

    predictions = get_predictions()

    if predictions.empty or predictions["error"].isna().all():
        return jsonify({"message": "No evaluated predictions yet"})

    mae = float(predictions["error"].dropna().mean())

    return jsonify({
        "Average_MAE": mae,
        "Total_Evaluated": int(predictions["error"].dropna().count())
    })


# ------------------------------------------------
# Forecast History (Evaluated)
# ------------------------------------------------
@app.route("/history")
def history():
    auto_evaluate()

    predictions = get_predictions().dropna()

    if predictions.empty:
        return jsonify([])

    return jsonify(
        predictions.tail(10).to_dict(orient="records")
    )


# ------------------------------------------------
# Live Temperature History
# ------------------------------------------------
@app.route("/live-history")
def live_history():
    weather = get_all_data()

    if weather.empty:
        return jsonify([])

    last_data = weather.tail(30)

    return jsonify(
        last_data[["timestamp", "temperature"]].to_dict(orient="records")
    )


if __name__ == "__main__":
    app.run(debug=True)
