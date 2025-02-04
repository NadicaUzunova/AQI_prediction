import os
import pandas as pd
import numpy as np

def process_data(input_filepath, output_filepath):
    """
    Procesira podatke iz vhodne CSV datoteke:
      - Pretvori stolpec 'date' v tip datetime (če še ni)
      - Odstrani podvajanje zapisov na podlagi stolpca 'date' (če za isti datum obstaja več zapisov, obdrži zadnji)
      - V numeričnih stolpcih (razen 'date') zapolni manjkajoče vrednosti z mediano vrednostjo
      - Doda stolpec 'category' na podlagi vrednosti 'eu_aqi'
      - Pretvori kategorične stolpce (razen 'date') v dummy spremenljivke
      - Preveri, ali podatki za določen datum že obstajajo; če ja, jih ne dodaja ponovno.
    """
    # Preberi vhodno CSV datoteko, stolpec 'date' pretvori v datetime
    df_new = pd.read_csv(input_filepath, parse_dates=["date"])
    
    # Odstrani podvajanje zapisov glede na 'date' (obdrži zadnji zapis za vsak datum)
    df_new = df_new.drop_duplicates(subset=["date"], keep="last")
    
    # Identificiramo numerične stolpce (razen 'date')
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    
    # Zapolnimo manjkajoče vrednosti v numeričnih stolpcih z mediano vrednostjo
    for col in numeric_cols:
        if df_new[col].isnull().any():
            median_val = df_new[col].median()
            df_new[col].fillna(median_val, inplace=True)
    
    # Dodajanje kategorije na podlagi 'eu_aqi'
    def categorize_aqi(aqi):
        if aqi <= 20:
            return "good"
        elif aqi <= 40:
            return "fair"
        elif aqi <= 60:
            return "moderate"
        elif aqi <= 80:
            return "poor"
        elif aqi <= 100:
            return "very poor"
        else:
            return "extremely poor"
    
    df_new["category"] = df_new["eu_aqi"].apply(categorize_aqi)
    
    # Poskrbi, da je mapa za output ustvarjena
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Če datoteka že obstaja, preveri in dodaj samo nove podatke
    if os.path.exists(output_filepath):
        existing_df = pd.read_csv(output_filepath, parse_dates=["date"])
        existing_dates = set(existing_df["date"].dt.date)
        df_new = df_new[~df_new["date"].dt.date.isin(existing_dates)]

        if df_new.empty:
            print("📢 Ni novih podatkov za dodajanje.")
        else:
            combined_df = pd.concat([existing_df, df_new]).drop_duplicates(subset=["date"], keep="last")
            combined_df.to_csv(output_filepath, index=False)
            print(f"✅ Dodano {len(df_new)} novih zapisov v: {output_filepath}")
    else:
        df_new.to_csv(output_filepath, index=False)
        print(f"✅ Prva shranitev podatkov v: {output_filepath}")

def main():
    input_filepath = os.path.join("data", "raw", "merged_data_raw.csv")
    output_filepath = os.path.join("data", "processed", "dataset.csv")
    process_data(input_filepath, output_filepath)

if __name__ == "__main__":
    main()