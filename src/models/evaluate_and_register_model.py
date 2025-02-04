import os
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, accuracy_score, f1_score
from dotenv import load_dotenv

# Nastavitev MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow_client = MlflowClient()

# Fiksne poti do testnih podatkov
TEST_DATA_PATH = "data/processed/test/test_data.csv"

def get_latest_model(model_name):
    """
    Pridobi zadnjo verzijo modela iz MLflow Model Registry, ki je v fazi 'None'.
    """
    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=["None"])
        if versions:
            return versions[0].version
    except Exception as e:
        print(f"⚠️ Napaka pri pridobivanju zadnje verzije modela: {e}")
        return None
    return None

def get_production_model(model_name):
    """
    Pridobi trenutno produkcijsko verzijo modela.
    """
    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].version
    except Exception as e:
        print(f"⚠️ Napaka pri pridobivanju produkcijskega modela: {e}")
        return None
    return None

def evaluate_regression_model(model_uri, X_test, y_test):
    """
    Izvede evalvacijo regresijskega modela (napoved PM10).
    """
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        print("❌ Napaka pri nalaganju regresijskega modela!")
        return None, None, None

    predictions = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    return mae, mse, evs

def evaluate_classification_model(model_uri, X_test, y_test):
    """
    Izvede evalvacijo klasifikacijskega modela (napoved kategorije).
    """
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        print("❌ Napaka pri nalaganju klasifikacijskega modela!")
        return None, None

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    return accuracy, f1

def main():
    print("📡 Nalagam testne podatke...")
    test_data = pd.read_csv(TEST_DATA_PATH, parse_dates=["date"])

    features = ["pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "temperature_2m",
                "relative_humidity_2m", "rain", "snowfall", "is_day"]
    target_regression = "pm10"
    target_classification = "category"

    X_test = test_data[features]
    y_test_reg = test_data[target_regression]
    y_test_class = test_data[target_classification]

    # Pridobimo zadnjo verzijo modelov
    latest_reg_version = get_latest_model("regression_model")
    latest_class_version = get_latest_model("classification_model")

    if not latest_reg_version or not latest_class_version:
        print("❌ Ni nove verzije modela za evalvacijo.")
        return

    latest_reg_uri = f"models:/regression_model/{latest_reg_version}"
    latest_class_uri = f"models:/classification_model/{latest_class_version}"

    # Evaluacija novih modelov
    latest_mae, latest_mse, latest_evs = evaluate_regression_model(latest_reg_uri, X_test, y_test_reg)
    latest_acc, latest_f1 = evaluate_classification_model(latest_class_uri, X_test, y_test_class)

    if latest_mae is None or latest_acc is None:
        return

    print(f"🔎 Novi regresijski model (verzija {latest_reg_version}) - MAE: {latest_mae:.3f}, MSE: {latest_mse:.3f}, EVS: {latest_evs:.3f}")
    print(f"🔎 Novi klasifikacijski model (verzija {latest_class_version}) - Accuracy: {latest_acc:.3f}, F1-score: {latest_f1:.3f}")

    # Pridobitev trenutnega produkcijskega modela
    prod_reg_version = get_production_model("regression_model")
    prod_class_version = get_production_model("classification_model")

    if not prod_reg_version or not prod_class_version:
        print("✅ Ni obstoječega produkcijskega modela. Novi model bo označen kot 'Production'.")
        mlflow_client.transition_model_version_stage("regression_model", latest_reg_version, stage="Production")
        mlflow_client.transition_model_version_stage("classification_model", latest_class_version, stage="Production")
        return

    prod_reg_uri = f"models:/regression_model/{prod_reg_version}"
    prod_class_uri = f"models:/classification_model/{prod_class_version}"

    prod_mae, prod_mse, prod_evs = evaluate_regression_model(prod_reg_uri, X_test, y_test_reg)
    prod_acc, prod_f1 = evaluate_classification_model(prod_class_uri, X_test, y_test_class)

    if prod_mae is None or prod_acc is None:
        return

    print(f"🔎 Produkcijski regresijski model (verzija {prod_reg_version}) - MAE: {prod_mae:.3f}, MSE: {prod_mse:.3f}, EVS: {prod_evs:.3f}")
    print(f"🔎 Produkcijski klasifikacijski model (verzija {prod_class_version}) - Accuracy: {prod_acc:.3f}, F1-score: {prod_f1:.3f}")

    # Primerjava regresijskih modelov
    if latest_mse < prod_mse and latest_evs > prod_evs:
        print("🚀 Novi regresijski model je boljši. Posodabljam produkcijski model.")
        mlflow_client.transition_model_version_stage("regression_model", latest_reg_version, stage="Production")
        mlflow_client.transition_model_version_stage("regression_model", prod_reg_version, stage="Archived")
    else:
        print("📉 Novi regresijski model ni boljši. Ostanemo pri trenutnem produkcijskem modelu.")
        mlflow_client.transition_model_version_stage("regression_model", latest_reg_version, stage="Archived")

    # Primerjava klasifikacijskih modelov
    if latest_acc > prod_acc and latest_f1 > prod_f1:
        print("🚀 Novi klasifikacijski model je boljši. Posodabljam produkcijski model.")
        mlflow_client.transition_model_version_stage("classification_model", latest_class_version, stage="Production")
        mlflow_client.transition_model_version_stage("classification_model", prod_class_version, stage="Archived")
    else:
        print("📉 Novi klasifikacijski model ni boljši. Ostanemo pri trenutnem produkcijskem modelu.")
        mlflow_client.transition_model_version_stage("classification_model", latest_class_version, stage="Archived")

if __name__ == "__main__":
    main()