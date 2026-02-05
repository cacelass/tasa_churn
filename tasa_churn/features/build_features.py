# credito/features/build_features.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tasa_churn.utils.paths import ARTIFACTS_DIR

def preprocess_data(df, target_col='y', save_artifacts=True):
    """
    Procesa los datos para entrenamiento y guarda los codificadores.
        - Limpieza: elimina duplicados, nulos y columnas irrelevantes.
        - Codificación: LabelEncoder para Gender, mapeo para Subscription Type y Contract Length.
        - Escalado: MinMaxScaler para todas las columnas numéricas.
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
    # LabelEncoder solo para Gender
    labelencoder_gender = LabelEncoder()
    X['Gender'] = labelencoder_gender.fit_transform(X['Gender'])

    # Mapear otras categóricas
    X['Subscription Type'] = X['Subscription Type'].str.title().map({'Basic':0, 'Standard':1, 'Premium':2})
    X['Contract Length']   = X['Contract Length'].str.title().map({'Monthly':0, 'Quarterly':1, 'Annual':2})

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        encoders = {
            "Gender": labelencoder_gender,
            "Subscription Type": {'Basic': 0, 'Standard': 1, 'Premium': 2},
            "Contract Length": {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}
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
        - Carga columnas, encoders y scaler.
        - Crea un DataFrame con los datos del usuario.
        - Aplica las mismas transformaciones que al entrenamiento.
        - Devuelve un array listo para predecir.
        user_data: dict con claves como  'Gender', 'Age', 'Subscription Type', etc.
    """
    # 1. Cargar artefactos
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    except FileNotFoundError:
        raise Exception("No se encontraron los archivos de entrenamiento. Entrena el modelo primero.")

    # 2. Crear DataFrame vacío con las columnas correctas
    df = pd.DataFrame(columns=columns)
    
    # 3. Llenar el DataFrame con los datos del usuario
    for col in columns:
        if col in user_data:
            df.loc[0, col] = user_data[col]
        else:
            raise ValueError(f"Falta la columna requerida: {col}")
        
    # 4. Aplicar encoders solo a columnas categóricas
    # Gender
    if 'Gender' in df.columns and 'Gender' in encoders:
        gender_encoder = encoders['Gender']
        df['Gender'] = gender_encoder.transform(df['Gender'].astype(str))
    
    # Subscription Type
    if 'Subscription Type' in df.columns and 'Subscription Type' in encoders:
        sub_map = encoders['Subscription Type']
        df['Subscription Type'] = df['Subscription Type'].astype(str).str.title().map(sub_map)
        if df['Subscription Type'].isna().any():
            df['Subscription Type'] = 0
    
    # Contract Length
    if 'Contract Length' in df.columns and 'Contract Length' in encoders:
        contract_map = encoders['Contract Length']
        df['Contract Length'] = df['Contract Length'].astype(str).str.title().map(contract_map)
        if df['Contract Length'].isna().any():
            df['Contract Length'] = 0

    # 5. Convertir todo a numérico
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 6. Aplicar scaler
    df_scaled = scaler.transform(df)
    
    return df_scaled