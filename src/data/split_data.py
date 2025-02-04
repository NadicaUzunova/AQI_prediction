import os
import pandas as pd

def split_data(input_path, output_train, output_test, test_size_ratio=0.1):
    """Razdeli podatke na train in test glede na časovne žige."""
    
    # Preverimo, ali datoteka obstaja
    if not os.path.exists(input_path):
        print(f"⚠️ Opozorilo: {input_path} ne obstaja. Preskakujem...")
        return
    
    df = pd.read_csv(input_path, parse_dates=["date"])

    if df.empty:
        print(f"⚠️ Opozorilo: {input_path} je prazna. Preskakujem...")
        return
    
    # Sortiramo podatke glede na čas (da ohranimo zaporedje)
    df = df.sort_values(by="date")
    
    # Določimo velikost testnega nabora
    test_size = max(1, int(len(df) * test_size_ratio))
    
    # Razdelimo na train in test
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    # Ustvarimo mape, če še ne obstajajo
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)

    # Shranimo podatke
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"✅ Podatki razdeljeni: Train ({len(train_df)}), Test ({len(test_df)})")

def main():
    # Fiksne poti
    input_path = "data/processed/dataset.csv"
    output_train = "data/processed/train/train_data.csv"
    output_test = "data/processed/test/test_data.csv"
    
    # Test size ratio (lahko prilagodimo)
    test_size_ratio = 0.1

    # Izvedemo delitev podatkov
    split_data(input_path, output_train, output_test, test_size_ratio)

if __name__ == "__main__":
    main()