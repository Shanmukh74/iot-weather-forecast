import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

FILE_PATH = "../data/weather_data.csv"

# -----------------------------
# 1. Load Dataset (NO HEADER SAFE)
# -----------------------------
df = pd.read_csv(FILE_PATH, header=None)

# Assign proper column names manually
df.columns = ["timestamp", "temperature", "humidity", "pressure", "rainfall"]

print("Total rows loaded:", len(df))

# Convert numeric columns properly
df["temperature"] = pd.to_numeric(df["temperature"])
df["humidity"] = pd.to_numeric(df["humidity"])
df["pressure"] = pd.to_numeric(df["pressure"])
df["rainfall"] = pd.to_numeric(df["rainfall"])

# -----------------------------
# 2. Create Lag Features
# -----------------------------
df["temp_lag_1"] = df["temperature"].shift(1)
df["temp_lag_2"] = df["temperature"].shift(2)
df["temp_lag_6"] = df["temperature"].shift(6)
df["temp_lag_12"] = df["temperature"].shift(12)

# -----------------------------
# 3. Create 10-Min Future Target
# -----------------------------
df["future_temperature"] = df["temperature"].shift(-120)

# Drop rows with NaN values (from lag + shift)
df = df.dropna()

print("Rows after lag + shift:", len(df))

# -----------------------------
# 4. Define Features
# -----------------------------
X = df[
    [
        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "temp_lag_1",
        "temp_lag_2",
        "temp_lag_6",
        "temp_lag_12",
    ]
]

y = df["future_temperature"]

# -----------------------------
# 5. Train/Test Split (Time-Series Safe)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 6. Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate Model
# -----------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Evaluation Results:")
print("MAE:", mae)
print("R2 Score:", r2)

# -----------------------------
# 8. Save Model
# -----------------------------
joblib.dump(model, "temperature_forecast_model.pkl")

print("Model saved successfully as temperature_forecast_model.pkl")
