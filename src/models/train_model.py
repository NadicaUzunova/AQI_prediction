import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
import argparse

# Nastavitev MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow_client = MlflowClient()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Fiksne poti do podatkov
TRAIN_DATA_PATH = "data/processed/train/train_data.csv"

def train_model():
    """Treniranje hibridnega modela za napovedovanje PM10 (regresija) in category (klasifikacija)."""
    print(f"🚀 Začenjam učenje modelov...")

    # Nalaganje podatkov
    df = pd.read_csv(TRAIN_DATA_PATH, parse_dates=["date"])

    # Odstranimo stolpec "date", ker ni uporaben za učenje
    df = df.drop(columns=["date"])

    # Določimo značilke in ciljne spremenljivke
    features = ["pm2_5", "carbon_monoxide", "carbon_dioxide", "uv_index", "temperature_2m",
                "relative_humidity_2m", "rain", "snowfall", "is_day"]
    target_regression = "pm10"
    target_classification = "category"

    # Preverimo, ali so vsi zahtevani stolpci prisotni
    required_columns = features + [target_regression, target_classification]
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Napaka: Manjkajo stolpci v {TRAIN_DATA_PATH}")
        return

    # Razdelimo podatke na regresijski in klasifikacijski model
    X = df[features]
    y_regression = df[target_regression]
    y_classification = df[target_classification]

    # Pretvorimo kategorije v numerične vrednosti
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    y_classification_encoded = encoder.fit_transform(y_classification.to_numpy().reshape(-1, 1))

    # Razdelimo na train/test sklope
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    _, _, y_train_class, y_test_class = train_test_split(X, y_classification_encoded, test_size=0.2, random_state=42)

    # Predprocesiranje podatkov
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), features)
    ])

    # Model za regresijo (napoved PM10)
    mlp_regressor = MLPRegressor(max_iter=500, random_state=42)

    pipeline_regression = Pipeline([
        ("preprocess", preprocessor),
        ("MLPR", mlp_regressor)
    ])

    # Model za klasifikacijo (napoved kategorije)
    mlp_classifier = MLPClassifier(max_iter=500, random_state=42)

    pipeline_classification = Pipeline([
        ("preprocess", preprocessor),
        ("MLPC", mlp_classifier)
    ])

    # Parametri za optimizacijo modelov
    param_grid_regression = {
        "MLPR__hidden_layer_sizes": [(32,), (16,)],
        "MLPR__learning_rate_init": [0.001, 0.01]
    }
    param_grid_classification = {
        "MLPC__hidden_layer_sizes": [(32,), (16,)],
        "MLPC__learning_rate_init": [0.001, 0.01]
    }

    print("🔎 Optimizacija hiperparametrov za regresijski model...")
    search_reg = GridSearchCV(pipeline_regression, param_grid_regression, cv=3, verbose=2, n_jobs=-1)
    
    print("🔎 Optimizacija hiperparametrov za klasifikacijski model...")
    search_class = GridSearchCV(pipeline_classification, param_grid_classification, cv=3, verbose=2, n_jobs=-1)

    with mlflow.start_run(run_name="Train_Hybrid_Model"):
        search_reg.fit(X_train, y_train_reg)
        search_class.fit(X_train, y_train_class)

        best_params_reg = search_reg.best_params_
        best_params_class = search_class.best_params_

        # Končni modeli z optimiziranimi parametri
        final_regressor = Pipeline([
            ("preprocess", preprocessor),
            ("MLPR", MLPRegressor(
                hidden_layer_sizes=best_params_reg["MLPR__hidden_layer_sizes"], 
                learning_rate_init=best_params_reg["MLPR__learning_rate_init"], 
                max_iter=1000, random_state=42))
        ])
        final_regressor.fit(X_train, y_train_reg)

        final_classifier = Pipeline([
            ("preprocess", preprocessor),
            ("MLPC", MLPClassifier(
                hidden_layer_sizes=best_params_class["MLPC__hidden_layer_sizes"], 
                learning_rate_init=best_params_class["MLPC__learning_rate_init"], 
                max_iter=1000, random_state=42))
        ])
        final_classifier.fit(X_train, y_train_class)

        # Ocene modelov
        train_score_reg = final_regressor.score(X_train, y_train_reg)
        test_score_reg = final_regressor.score(X_test, y_test_reg)
        train_score_class = final_classifier.score(X_train, y_train_class)
        test_score_class = final_classifier.score(X_test, y_test_class)

        print(f"✅ Regresijski model: Train Score: {train_score_reg:.3f}, Test Score: {test_score_reg:.3f}")
        print(f"✅ Klasifikacijski model: Train Score: {train_score_class:.3f}, Test Score: {test_score_class:.3f}")

        # Logiranje parametrov in metrik v MLflow
        mlflow.log_param("best_hidden_layer_sizes_reg", best_params_reg["MLPR__hidden_layer_sizes"])
        mlflow.log_param("best_learning_rate_reg", best_params_reg["MLPR__learning_rate_init"])
        mlflow.log_metric("train_score_reg", train_score_reg)
        mlflow.log_metric("test_score_reg", test_score_reg)

        mlflow.log_param("best_hidden_layer_sizes_class", best_params_class["MLPC__hidden_layer_sizes"])
        mlflow.log_param("best_learning_rate_class", best_params_class["MLPC__learning_rate_init"])
        mlflow.log_metric("train_score_class", train_score_class)
        mlflow.log_metric("test_score_class", test_score_class)

        # Shranjevanje modelov v MLflow
        mlflow.sklearn.log_model(final_regressor, "regression_model")
        mlflow.sklearn.log_model(final_classifier, "classification_model")

        print("📌 Modeli so shranjeni in registrirani v MLflow!")

def main():
    train_model()

if __name__ == "__main__":
    main()