import pandas as pd
import requests

# ================== CONFIG ==================
INPUT_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\Energy\\long_data_.csv"
OUTPUT_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\Energy\\updated_weather_data.xlsx"

DATE_COL = "Dates"        # exact column name in your CSV
LAT_COL = "latitude"
LON_COL = "longitude"
# ============================================


def fetch_weather_for_location(lat, lon, start_date, end_date):
    """
    Fetch historical daily weather for a location over a date range
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean"
        "&timezone=auto"
    )

    response = requests.get(url, timeout=30)
    data = response.json()

    if "daily" not in data:
        return pd.DataFrame()

    return pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temperature_celsius": data["daily"]["temperature_2m_mean"],
        "humidity_percent": data["daily"]["relative_humidity_2m_mean"],
    })


def main():
    # ---------- Load Dataset ----------
    df = pd.read_csv(INPUT_FILE)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse dates (DD/MM/YYYY HH:MM:SS)
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        dayfirst=True,
        errors="coerce"
    )

    # Drop rows with invalid dates
    df = df.dropna(subset=[DATE_COL])

    print(f"📊 Total rows after cleaning: {len(df)}")

    weather_frames = []

    # ---------- Group by location ----------
    grouped = df.groupby([LAT_COL, LON_COL])
    print(f"📍 Unique locations: {len(grouped)}")

    for idx, ((lat, lon), group) in enumerate(grouped, start=1):
        start_date = group[DATE_COL].min().strftime("%Y-%m-%d")
        end_date = group[DATE_COL].max().strftime("%Y-%m-%d")

        print(
            f"🌦️ Fetching weather {idx}/{len(grouped)} "
            f"for ({lat}, {lon}) from {start_date} to {end_date}"
        )

        weather_df = fetch_weather_for_location(lat, lon, start_date, end_date)

        if weather_df.empty:
            continue

        weather_df[LAT_COL] = lat
        weather_df[LON_COL] = lon

        weather_frames.append(weather_df)

    # ---------- Combine all weather ----------
    weather_all = pd.concat(weather_frames, ignore_index=True)

    # ---------- Merge weather with original data ----------
    df["merge_date"] = df[DATE_COL].dt.normalize()

    final_df = df.merge(
        weather_all,
        left_on=["merge_date", LAT_COL, LON_COL],
        right_on=["date", LAT_COL, LON_COL],
        how="left"
    )

    # Cleanup helper columns
    final_df.drop(columns=["merge_date", "date"], inplace=True)

    # ---------- Save output ----------
    final_df.to_csv("C:/Users/HP/OneDrive/Desktop/Energy/updated_weather_data.csv", index=False)


    print("✅ DONE!")
    print(f"📁 Output file created at:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
