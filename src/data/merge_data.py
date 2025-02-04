import os
import pandas as pd

def merge_data(aqi_filepath, weather_filepath, output_filepath):
    """
    Prebere CSV datoteki z AQI in vremenskimi podatki,
    združi podatke na podlagi stolpca 'date' (če se datum ujema, se prilepijo stolpci iz weather)
    in shrani združen rezultat kot CSV.
    """
    if not os.path.exists(aqi_filepath):
        print(f"⚠️ AQI datoteka ne obstaja: {aqi_filepath}")
        return

    if not os.path.exists(weather_filepath):
        print(f"⚠️ Weather datoteka ne obstaja: {weather_filepath}")
        return

    df_aqi = pd.read_csv(aqi_filepath, parse_dates=["date"])
    df_weather = pd.read_csv(weather_filepath, parse_dates=["date"])

    df_merged = pd.merge(df_aqi, df_weather, on="date", how="inner")
    df_merged.sort_values("date", inplace=True)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    if os.path.exists(output_filepath):
        existing_df = pd.read_csv(output_filepath, parse_dates=["date"])
        existing_dates = set(existing_df["date"].dt.date)
        df_new = df_merged[~df_merged["date"].dt.date.isin(existing_dates)]

        if df_new.empty:
            print("📢 Ni novih podatkov za združitev.")
        else:
            combined_df = pd.concat([existing_df, df_new]).drop_duplicates(subset=["date"], keep="last")
            combined_df.to_csv(output_filepath, index=False)
            print(f"✅ Dodano {len(df_new)} novih zapisov v: {output_filepath}")
    else:
        df_merged.to_csv(output_filepath, index=False)
        print(f"✅ Prva shranitev podatkov v: {output_filepath}")

def main():
    aqi_filepath = os.path.join("data", "raw", "aqi", "aqi_data.csv")
    weather_filepath = os.path.join("data", "raw", "weather", "weather_data.csv")
    output_filepath = os.path.join("data", "raw", "merged_data_raw.csv")
    
    merge_data(aqi_filepath, weather_filepath, output_filepath)

if __name__ == "__main__":
    main()