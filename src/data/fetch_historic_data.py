import os
from datetime import datetime
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

def fetch_aqi_data():
    """
    Pridobi podatke o kakovosti zraka prek Open-Meteo API-ja.
    
    Uporablja endpoint:
      https://air-quality-api.open-meteo.com/v1/air-quality
    
    Parametri:
      - latitude: 46.55
      - longitude: 15.64
      - hourly: ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "european_aqi"]
      - past_days: 92
      - forecast_days: 1
    
    Vrne:
      pandas DataFrame s stolpci:
        - date (časovna os)
        - pm10, pm2_5, carbon_monoxide, carbon_dioxide, uv_index
    """
    # Nastavimo session z cache in retry (cache expira čez eno uro)
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    
    # Ustvarimo Open-Meteo API klienta
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 46.55,
        "longitude": 15.64,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "european_aqi"],
        "past_days": 92,
        "forecast_days": 1
    }
    
    responses = openmeteo.weather_api(url, params=params)
    # Predpostavljamo, da je rezultat seznam odgovorov; vzamemo prvi
    response = responses[0]
    
    # Izpišemo nekaj metapodatkov
    print(f"[AQI] Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"[AQI] Elevation: {response.Elevation()} m asl")
    print(f"[AQI] Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"[AQI] UTC offset: {response.UtcOffsetSeconds()} s")
    
    # Pridobimo podatke iz odseka "Hourly"
    hourly = response.Hourly()
    # Vrstni red spremenljivk mora biti enak, kot je naveden v parametrih "hourly"
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_carbon_dioxide = hourly.Variables(3).ValuesAsNumpy()
    hourly_uv_index = hourly.Variables(4).ValuesAsNumpy()
    hourly_european_aqi = hourly.Variables(5).ValuesAsNumpy()
    
    # Ustvarimo časovno os – uporabimo čas začetka, konec in interval iz odgovora
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly_pm10,
        "pm2_5": hourly_pm2_5,
        "carbon_monoxide": hourly_carbon_monoxide,
        "carbon_dioxide": hourly_carbon_dioxide,
        "uv_index": hourly_uv_index,
        "eu_aqi": hourly_european_aqi
    }
    
    df_aqi = pd.DataFrame(data=hourly_data)
    return df_aqi

def fetch_weather_data():
    """
    Pridobi zgodovinske vremenske podatke prek arhivnega Open-Meteo API-ja.
    
    Uporablja endpoint:
      https://archive-api.open-meteo.com/v1/archive
    
    Parametri:
      - latitude: 46.55
      - longitude: 15.64
      - start_date: "2024-10-01"
      - end_date: "2025-02-04"
      - hourly: ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "is_day"]
    
    Vrne:
      pandas DataFrame s stolpci:
        - date (časovna os)
        - temperature_2m, relative_humidity_2m, rain, snowfall, is_day
    """
    # Nastavimo session z cache in retry; tukaj nastavimo cache brez poteka (expire_after=-1)
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    
    # Ustvarimo Open-Meteo API klienta
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 46.55,
        "longitude": 15.64,
        "start_date": "2024-10-01",
        "end_date": "2025-02-04",
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "is_day"]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Izpišemo metapodatke
    print(f"[Weather] Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"[Weather] Elevation: {response.Elevation()} m asl")
    print(f"[Weather] Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"[Weather] UTC offset: {response.UtcOffsetSeconds()} s")
    
    # Pridobimo podatke iz odseka "Hourly"
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(4).ValuesAsNumpy()
    
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m,
        "relative_humidity_2m": hourly_relative_humidity_2m,
        "rain": hourly_rain,
        "snowfall": hourly_snowfall,
        "is_day": hourly_is_day
    }
    
    df_weather = pd.DataFrame(data=hourly_data)
    return df_weather

def save_dataframe(df, directory, prefix):
    """
    Shrani podani pandas DataFrame v CSV datoteko v navedeni mapi.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{prefix}.csv")
    df.to_csv(filepath, index=False)
    print(f"Podatki so shranjeni v: {filepath}")

def main():
    print("Pridobivanje AQI podatkov...")
    df_aqi = fetch_aqi_data()
    print("Pridobljeni AQI podatki:")
    print(df_aqi.head(), "\n")
    
    print("Pridobivanje vremenskih podatkov...")
    df_weather = fetch_weather_data()
    print("Pridobljeni vremenski podatki:")
    print(df_weather.head(), "\n")
    
    # Shrani podatke v ustrezne mape:
    aqi_dir = os.path.join("data", "raw", "aqi")
    weather_dir = os.path.join("data", "raw", "weather")
    
    save_dataframe(df_aqi, aqi_dir, "aqi_data")
    save_dataframe(df_weather, weather_dir, "weather_data")

if __name__ == "__main__":
    main()