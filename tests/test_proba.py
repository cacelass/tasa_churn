# tests/test_proba.py
import pytest
import numpy as np
import pandas as pd
from tasa_churn.features.build_features import preprocess_data, process_input

@pytest.fixture
def sample_df():
    """DataFrame de ejemplo similar al dataset original."""
    data = {
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Subscription Type": ["Basic", "Premium"],
        "Contract Length": ["Monthly", "Annual"],
        "Usage Frequency": [5, 3],
        "Last Interaction": [10, 15],
        "Total Spend": [200, 500],
        "Age": [35, 50],
        "Tenure": [24, 36],
        "Support Calls": [2, 1],
        "Payment Delay": [3, 5],
        "Churn": [0, 1],
    }
    return pd.DataFrame(data)

def test_process_input(sample_df):
    """Comprueba que process_input transforma correctamente datos de usuario"""
    # 1. Generamos artefactos con preprocess_data
    preprocess_data(sample_df.copy(), target_col="Churn", save_artifacts=True)

    # 2. Input simulado de usuario (puede estar fuera de rango)
    user_input = {
        "Gender": "Male",
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Age": 30,       # fuera del rango de entrenamiento
        "Tenure": 10,    # fuera del rango de entrenamiento
        "Support Calls": 0,
        "Payment Delay": 2
    }

    # 3. Procesamos input
    processed = process_input(user_input)

    # 4. Validaciones
    assert isinstance(processed, np.ndarray), "El output debe ser un array numpy"
