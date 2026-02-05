import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from tasa_churn.features.build_features import preprocess_data, process_input
from tasa_churn.utils.paths import ARTIFACTS_DIR

@pytest.fixture
def sample_df():
    data = {
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Subscription Type": ["Basic", "Premium"],
        "Contract Length": ["Monthly", "Annual"],
        "Usage Frequency": [5, 10],
        "Last Interaction": [3, 2],
        "Total Spend": [200, 400],
        "Age": [35, 42],
        "Tenure": [24, 36],
        "Support Calls": [2, 1],
        "Payment Delay": [3, 0],
        "Churn": [1, 0]
    }
    return pd.DataFrame(data)


def test_preprocess_columns_and_encoding(sample_df):
    X_train, X_test, y_train, y_test = preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)

    # Columnas irrelevantes eliminadas
    for col in ["CustomerID", "Usage Frequency", "Last Interaction", "Total Spend"]:
        assert col not in X_train.columns

    # Gender está codificado con LabelEncoder
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    assert set(X_train['Gender']).issubset({0, 1})
    assert hasattr(encoders['Gender'], 'classes_')

    # Diccionarios no deben cambiarse
    assert set(X_train['Subscription Type']).issubset({0, 2})
    assert set(X_train['Contract Length']).issubset({0, 2})

    # Escalado entre 0 y 1
    assert np.all((X_train >= 0) & (X_train <= 1))

def test_artifacts_saved(sample_df):
    preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)
    # Verificar que los artefactos existen
    assert Path(ARTIFACTS_DIR / "columns.joblib").exists()
    assert Path(ARTIFACTS_DIR / "encoders.joblib").exists()
    assert Path(ARTIFACTS_DIR / "scaler.joblib").exists()

def test_process_input_basic(sample_df):
    preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)
    user_input = {
        "Gender": "Male",
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Age": 30,
        "Tenure": 10,
        "Support Calls": 0,
        "Payment Delay": 2
    }
    processed = process_input(user_input)
    # Devuelve numpy array
    assert isinstance(processed, np.ndarray)
    # Tiene tantas columnas como el DataFrame entrenado
    columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
    assert processed.shape[1] == len(columns)

def test_process_input_scaling(sample_df):
    preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)
    user_input = {
        "Gender": "Female",
        "Subscription Type": "Premium",
        "Contract Length": "Annual",
        "Age": 42,
        "Tenure": 36,
        "Support Calls": 1,
        "Payment Delay": 0
    }
    processed = process_input(user_input)
    # Comprobar que las columnas numéricas están escaladas entre 0 y 1
    assert np.all((processed >= 0) & (processed <= 1))

def test_process_input_unknown_category(sample_df):
    preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)
    user_input = {
        "Gender": "Other",  # No estaba en entrenamiento
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Age": 25,
        "Tenure": 5,
        "Support Calls": 0,
        "Payment Delay": 1
    }
    with pytest.raises(ValueError):
        process_input(user_input)

def test_integration_consistency(sample_df):
    X_train, X_test, y_train, y_test = preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)
    user_input = {
        "Gender": "Male",
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Age": 30,
        "Tenure": 10,
        "Support Calls": 0,
        "Payment Delay": 2
    }
    processed = process_input(user_input)
    # La forma debe coincidir con columnas de X_train
    assert processed.shape[1] == X_train.shape[1]
