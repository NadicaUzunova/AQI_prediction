import os
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime

def fetch_aqi_data():
    """
    Pridobi sveže podatke o kakovosti zraka prek Open-Meteo API-ja.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 46.55,
        "longitude": 15.64,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "european_aqi"],
        "past_days": 1,
        "forecast_days": 1
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
        "carbon_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "uv_index": hourly.Variables(4).ValuesAsNumpy(),
        "eu_aqi": hourly.Variables(5).ValuesAsNumpy()
    }
    df_aqi = pd.DataFrame(data=hourly_data)
    return df_aqi

def fetch_weather_data():
    """
    Pridobi sveže zgodovinske vremenske podatke prek Open-Meteo API-ja.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    params = {
        "latitude": 46.55,
        "longitude": 15.64,
        "start_date": current_date,
        "end_date": current_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "is_day"]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "rain": hourly.Variables(2).ValuesAsNumpy(),
        "snowfall": hourly.Variables(3).ValuesAsNumpy(),
        "is_day": hourly.Variables(4).ValuesAsNumpy()
    }
    df_weather = pd.DataFrame(data=hourly_data)
    return df_weather

def update_or_append_csv(df_new, filepath):
    """
    Preveri, ali podatki za določen datum že obstajajo v CSV datoteki.
    - Če datum že obstaja, podatkov ne doda ponovno.
    - Če datum ne obstaja, doda nov zapis.
    """
    df_new["date"] = pd.to_datetime(df_new["date"])

    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath, parse_dates=["date"])
        existing_dates = set(existing_df["date"].dt.date)

        df_new_filtered = df_new[~df_new["date"].dt.date.isin(existing_dates)]
        
        if df_new_filtered.empty:
            print(f"📢 Ni novih podatkov za {filepath}.")
        else:
            combined_df = pd.concat([existing_df, df_new_filtered]).drop_duplicates(subset=["date"], keep="last")
            combined_df.to_csv(filepath, index=False)
            print(f"✅ Dodano {len(df_new_filtered)} novih zapisov v: {filepath}")
    else:
        df_new.to_csv(filepath, index=False)
        print(f"✅ Prva shranitev podatkov v: {filepath}")

def main():
    print("📡 Pridobivanje svežih AQI podatkov...")
    df_aqi = fetch_aqi_data()
    print("✅ Sveži AQI podatki (prvih 5 vrstic):")
    print(df_aqi.head(), "\n")

    print("📡 Pridobivanje svežih vremenskih podatkov...")
    df_weather = fetch_weather_data()
    print("✅ Sveži vremenski podatki (prvih 5 vrstic):")
    print(df_weather.head(), "\n")

    aqi_filepath = os.path.join("data", "raw", "aqi", "aqi_data.csv")
    weather_filepath = os.path.join("data", "raw", "weather", "weather_data.csv")

    update_or_append_csv(df_aqi, aqi_filepath)
    update_or_append_csv(df_weather, weather_filepath)

if __name__ == "__main__":
    main()