import random
import pandas as pd
import os
import time
import math
from datetime import datetime

FILE_PATH = "../data/weather_data.csv"

# Keep track of previous temperature for smooth transitions
previous_temp = 28

def generate_weather_data(step):
    global previous_temp

    # Simulate day-night temperature pattern using sine wave
    hour_fraction = (step % 8640) / 8640   # 8640 steps ≈ 12 hours (5 sec interval)
    base_temp = 28 + 6 * math.sin(2 * math.pi * hour_fraction)

    # Small natural variation
    noise = random.uniform(-0.5, 0.5)

    # Rainfall event occasionally
    rainfall = random.choice([0, 0, 0, 0, random.uniform(0, 10)])

    # If raining, temperature slightly decreases
    if rainfall > 0:
        base_temp -= random.uniform(1, 2)

    # Smooth temperature transition
    temperature = previous_temp + (base_temp - previous_temp) * 0.1 + noise
    previous_temp = temperature

    # Humidity inversely related to temperature
    humidity = 70 - (temperature - 28) * 2 + random.uniform(-3, 3)

    # Pressure slight slow variation
    pressure = 1005 + random.uniform(-5, 5)

    data = {
        "timestamp": datetime.now(),
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "pressure": round(pressure, 2),
        "rainfall": round(rainfall, 2)
    }

    df = pd.DataFrame([data])

    if not os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, index=False)
    else:
        df.to_csv(FILE_PATH, mode='a', header=False, index=False)

    print("Weather data added")


if __name__ == "__main__":
    step = 0
    while True:
        generate_weather_data(step)
        step += 1
        time.sleep(5)
