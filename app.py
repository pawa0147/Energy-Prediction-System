from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import requests
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ==========================
# Load Model
# ==========================
model = joblib.load(open("ridge_model.pkl", "rb"))

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("energy_with_features.csv")
df["Dates"] = pd.to_datetime(df["Dates"], format="mixed", dayfirst=True, errors="coerce")
df = df.dropna(subset=["Dates"])

# ==========================
# Fetch Weather
# ==========================
def fetch_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        temperature = data["current_weather"]["temperature"]
        humidity = data["hourly"]["relative_humidity_2m"][-1]

        return temperature, humidity
    except:
        return 30, 60


# ==========================
# Dashboard
# ==========================
@app.route("/")
def dashboard():

    states = sorted(df["States"].unique())
    state = states[0]

    state_data = df[df["States"] == state].sort_values("Dates")
    latest_row = state_data.iloc[-1]

    lat = latest_row["latitude"]
    lon = latest_row["longitude"]

    temperature, humidity = fetch_weather(lat, lon)

    now = datetime.now()
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    day = now.day
    month = now.month
    quarter = (month - 1) // 3 + 1

    features = np.array([[ 
        lat,
        lon,
        temperature,
        humidity,
        day_of_week,
        is_weekend,
        day,
        month,
        quarter,
        latest_row["Lag1"],
        latest_row["Lag7"],
        latest_row["Lag14"],
        latest_row["Rolling7"],
        latest_row["Rolling14"],
        latest_row["Rolling30"],
        latest_row["RollingStd7"]
    ]])

    prediction = model.predict(features)[0]
    today_usage = round(latest_row["Usage"], 2)

    return render_template(
        "index.html",
        states=states,
        today_usage=today_usage,
        prediction=round(prediction, 2),
        temperature=round(temperature, 2),
        humidity=round(humidity, 2)
    )


# ==========================
# Monthly Data API
# ==========================
@app.route("/monthly_data/<state>")
def monthly_data(state):

    state_data = df[df["States"] == state].copy()

    # Extract month number (1–12)
    state_data["month"] = state_data["Dates"].dt.month

    monthly_avg = (
        state_data
        .groupby("month")["Usage"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    # Convert month numbers to names
    monthly_avg["month"] = monthly_avg["month"].apply(
        lambda x: datetime(1900, x, 1).strftime("%b")
    )

    return jsonify({
        "months": monthly_avg["month"].tolist(),
        "usage": monthly_avg["Usage"].round(2).tolist()
    })
# ==========================
# Behavioral Data API
# ==========================
@app.route("/behavioral_data/<state>")
def behavioral_data(state):

    state_data = df[df["States"] == state].copy()
    state_data["day_of_week"] = state_data["Dates"].dt.weekday

    weekday_avg = state_data[state_data["day_of_week"] < 5]["Usage"].mean()
    weekend_avg = state_data[state_data["day_of_week"] >= 5]["Usage"].mean()

    return jsonify({
        "weekday": round(float(weekday_avg), 2),
        "weekend": round(float(weekend_avg), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)