import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from dotenv import load_dotenv
from datetime import datetime
from pymongo import MongoClient

# Nastavitev okolja
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Povezava z MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["aqiPredictions"]
collection = db["predictions"]

# Fiksna pot do testnih podatkov
INPUT_DATA_PATH = "data/processed/test/test_data.csv"

def save_predictions_to_mongo(input_data, predictions_reg, predictions_class, model_reg, model_class):
    """Shrani napovedi za PM10 in category v MongoDB."""
    timestamp = datetime.now().isoformat()
    documents = []

    for i in range(len(predictions_reg)):
        doc = {
            "timestamp": timestamp,
            "model_regression": model_reg,
            "model_classification": model_class,
            "features": {col: float(input_data.iloc[i][col]) for col in input_data.columns},
            "predicted_pm10": float(predictions_reg[i]),
            "predicted_category": predictions_class[i]  # Ostane kategorična vrednost
        }
        documents.append(doc)

    collection.insert_many(documents)
    print(f"✅ Napovedi shranjene v MongoDB.")

def load_production_model(model_name):
    """Naloži najnovejši 'Production' model iz MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()
    models = client.get_latest_versions(model_name, stages=["Production"])

    if not models:
        print(f"❌ Ni modelov v 'Production' za {model_name}")
        return None

    model_uri = f"models:/{model_name}/{models[0].version}"
    print(f"✅ Nalagam model {model_name} (verzija {models[0].version})...")
    return mlflow.sklearn.load_model(model_uri), models[0].version

def predict():
    """Izvede napovedi s produkcijskim modelom in jih shrani v MongoDB."""
    # Nalaganje modelov
    model_reg, version_reg = load_production_model("regression_model")
    model_class, version_class = load_production_model("classification_model")

    if model_reg is None or model_class is None:
        return

    # Nalaganje podatkov
    df = pd.read_csv(INPUT_DATA_PATH)
    features = ["pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "temperature_2m",
                "relative_humidity_2m", "rain", "snowfall", "is_day"]

    if not all(f in df.columns for f in features):
        print(f"❌ Manjkajoče značilke v {INPUT_DATA_PATH}")
        return

    X = df[features]

    # Napovedi
    predictions_reg = model_reg.predict(X)
    predictions_class = model_class.predict(X)  # Kategorije ostanejo nespremenjene

    # Shrani napovedi v MongoDB
    save_predictions_to_mongo(df[features], predictions_reg, predictions_class, version_reg, version_class)

    print(f"✅ Napovedi za PM10 in kategorijo uspešno izvedene.")

if __name__ == "__main__":
    predict()