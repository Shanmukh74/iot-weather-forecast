import random
import time
import math
import requests

WRITE_API_KEY = "OVUERDES5QNVRAK9"

previous_temp = 28

def generate_weather_data(step):
    global previous_temp

    hour_fraction = (step % 8640) / 8640
    base_temp = 28 + 6 * math.sin(2 * math.pi * hour_fraction)

    noise = random.uniform(-0.5, 0.5)

    rainfall = random.choice([0, 0, 0, 0, random.uniform(0, 10)])

    if rainfall > 0:
        base_temp -= random.uniform(1, 2)

    temperature = previous_temp + (base_temp - previous_temp) * 0.1 + noise
    previous_temp = temperature

    humidity = 70 - (temperature - 28) * 2 + random.uniform(-3, 3)
    pressure = 1005 + random.uniform(-5, 5)

    url = "https://api.thingspeak.com/update"

    payload = {
        "api_key": WRITE_API_KEY,
        "field1": round(temperature, 2),
        "field2": round(humidity, 2),
        "field3": round(pressure, 2),
        "field4": round(rainfall, 2)
    }

    response = requests.post(url, data=payload)

    print("Sent:", payload, "Response:", response.text)


if __name__ == "__main__":
    step = 0
    while True:
        generate_weather_data(step)
        step += 1
        time.sleep(15)  # IMPORTANT