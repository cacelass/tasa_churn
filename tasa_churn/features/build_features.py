# credito/features/build_features.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tasa_churn.utils.paths import ARTIFACTS_DIR

def preprocess_data(df, target_col='y', save_artifacts=True):
    """
    Procesa los datos para entrenamiento y guarda los codificadores.
    """
    print("--> Preprocesando datos de entrenamiento...")
    
    # 1. Limpieza
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Borrar CustomerID
    df.drop(columns=['CustomerID'], inplace=True)
    #Borrar Usage Frequency
    #Un cliente que hace churn → deja de usar el servicio Entonces su frecuencia baja brutalmente El modelo aprende: “frecuencia ≈ 0 ⇒ churn"
    df.drop(columns=['Usage Frequency'], inplace=True)
    # Borrar Last Interaction Un cliente que churnea → deja de interactuar Last Interaction se vuelve enorme El modelo lo adivina sin esfuerzo
    df.drop(columns=['Last Interaction'], inplace=True)
    # Borrar total spend El gasto total está muy correlacionado con la duración del contrato (tenure). El modelo puede aprender esta relación en lugar de aprender patrones reales relacionados con el churn.
    df.drop(columns=['Total Spend'], inplace=True)
    # Borrar el target de el dataset de test
    df.drop(columns=['Churn'], inplace=True)

    # Separar X e y
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # Caso para predicción sin target
        X = df
        y = None

    # Guardamos el orden de las columnas para pedirselas al usuario luego
    if save_artifacts:
        joblib.dump(X.columns.tolist(), ARTIFACTS_DIR / "columns.joblib")

   # 2. Categóricas
    labelencoder_X = LabelEncoder()

    # -------------------------- Preparación train -------------------------
    df['Gender'] = (df['Gender'].str.upper())

    df['Gender'] = labelencoder_X.fit_transform(df['Gender'])

    df['Subscription Type'] = (df['Subscription Type'].str.upper().map({'BASIC': 0, 'STANDARD': 1, 'PREMIUM': 2}))

    df['Contract Length'] = (df['Contract Length'].str.upper().map({'MONTHLY': 0, 'QUARTERLY': 1, 'ANNUAL': 2}))

    if save_artifacts:
        encoders = {
            "Gender": labelencoder_X,
            "Subscription Type": {'BASIC': 0, 'STANDARD': 1, 'PREMIUM': 2},
            "Contract Length": {'MONTHLY': 0, 'QUARTERLY': 1, 'ANNUAL': 2}
        }
        joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")

    # 3. Escalado
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    if save_artifacts:
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    # Retorno
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X_scaled

def process_input(user_data):
    """
    Toma un diccionario con los datos del usuario y los transforma
    usando los artefactos guardados.
    """
    # 1. Cargar artefactos
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    except FileNotFoundError:
        raise Exception("No se encontraron los archivos de entrenamiento. Entrena el modelo primero.")

    # 2. Crear DataFrame con las columnas correctas
    df = pd.DataFrame([user_data])
    
    # Asegurar que el orden de columnas es el mismo que en el entrenamiento
    df = df[columns]

    # 3. Aplicar Encoders (Categorías)
    for col, le in encoders.items():
        # Manejo básico de errores si el usuario pone algo desconocido
        try:
            df[col] = le.transform(df[col].astype(str))
        except ValueError:
            # Si el valor no existe (ej: Trabajo='Youtuber'), asignamos un valor por defecto o fallamos
            print(f"Advertencia: El valor '{df[col].iloc[0]}' en '{col}' no se vio en el entrenamiento.")
            df[col] = 0 # Asignamos 0 por defecto (o podrías lanzar error)

    # 4. Aplicar Scaler (Numéricos)
    df_scaled = scaler.transform(df)
    
    return df_scaled