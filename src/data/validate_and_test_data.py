import os
import shutil
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from great_expectations.data_context import DataContext
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import PandasDataset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Fiksne poti do podatkov
REFERENCE_DATA_PATH = "data/processed/train/train_data.csv"
CURRENT_DATA_PATH = "data/processed/test/test_data.csv"

def load_data(file_path):
    """Naloži CSV podatke v pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["date"])
    else:
        print(f"⚠️ Datoteka ne obstaja: {file_path}")
        return None

def validate_data(data, suite_name):
    """Validacija podatkov s Great Expectations."""
    print(f"🔹 Začenjam validacijo podatkov...")

    context = DataContext()
    
    try:
        suite = context.get_expectation_suite(suite_name)
    except:
        suite = ExpectationSuite(suite_name)
        context.add_expectation_suite(expectation_suite=suite)

    # Dodamo pravila validacije
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "date"}))
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "pm10"}))
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_values_to_be_between", kwargs={"column": "pm10", "min_value": 0, "max_value": 500}))
    suite.add_expectation(ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "category"}))

    dataset = PandasDataset(data)
    results = dataset.validate(expectation_suite=suite, only_return_failures=False)

    if not results.success:
        print(f"❌ Validacija ni uspela! Napake: {results}")
        exit(1)
    else:
        print(f"✅ Validacija uspešna!")

def test_data_drift(reference_data, current_data):
    """Izvede Evidently test za odkrivanje data drift-a."""
    print(f"🔹 Testiranje data drift-a...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_results = report.as_dict()
    if drift_results["metrics"][0]["result"]["dataset_drift"]:
        print(f"❌ Opozorilo: Zaznan data drift!")
    else:
        print(f"✅ Ni zaznanega data drift-a.")

def kolmogorov_smirnov_test(reference_data, current_data):
    """Kolmogorov-Smirnov test za preverjanje sprememb v distribuciji podatkov."""
    print(f"🔹 Izvajanje Kolmogorov-Smirnov testa...")

    numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        ks_stat, ks_p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())

        if ks_p_value < 0.05:
            print(f"❌ KS test ni uspešen za stolpec: {col} (p-value={ks_p_value:.5f})")
            exit(1)
        else:
            print(f"✅ KS test uspešen za stolpec: {col} (p-value={ks_p_value:.5f})")

def main():
    print("📡 Nalagam referenčne in trenutne podatke...")
    
    reference_data = load_data(REFERENCE_DATA_PATH)
    current_data = load_data(CURRENT_DATA_PATH)

    if reference_data is None or current_data is None:
        print("❌ Manjkajo podatki! Prekinjam validacijo.")
        exit(1)

    # Validacija podatkov
    validate_data(current_data, "aqi_validation")

    # Test data drift
    test_data_drift(reference_data, current_data)

    # Kolmogorov-Smirnov test
    kolmogorov_smirnov_test(reference_data, current_data)

    # Posodobitev referenčnih podatkov
    print(f"📌 Posodabljam referenčne podatke...")
    shutil.copy(CURRENT_DATA_PATH, REFERENCE_DATA_PATH)

    print("🚀 Validacija in testiranje uspešno zaključeno!")

if __name__ == "__main__":
    main()