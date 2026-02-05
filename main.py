import sys
import os
import joblib
import pandas as pd

# Importamos las rutas y funciones de tu proyecto
from tasa_churn.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from tasa_churn.data.make_dataset import load_data
from tasa_churn.features.build_features import preprocess_data, process_input
from tasa_churn.models.train_model import train_models
from tasa_churn.models.predict_model import evaluate_models

# Nombre del modelo que vamos a usar
MODEL_NAME = "RandomForest.joblib" 

def check_is_trained():
    """Verifica si existen el modelo y los archivos de traducción (encoders)."""
    model_path = MODELS_DIR / MODEL_NAME
    # Verificamos también que existan los codificadores (necesarios para traducir inputs)
    artifacts_exist = (ARTIFACTS_DIR / "encoders.joblib").exists()
    return model_path.exists() and artifacts_exist

def ask_user_data():
    """
    Pide los datos al usuario de forma interactiva y SEGURA.
    No permite avanzar si el dato no es válido.
    """
    print("\n" + "="*40)
    print("   RIESGO DE CHURN - PREDICCIÓN")
    print("="*40)
    
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    except FileNotFoundError:
        print(" Error: Faltan archivos de entrenamiento.")
        print("   Por favor, borra la carpeta 'models' y ejecuta de nuevo para re-entrenar.")
        sys.exit(1)
    
    user_data = {}
    
    for col in columns:
        # --- CASO A: Columna de TEXTO (Categoría) ---
        if col in encoders:
            encoder = encoders[col]
            
            # Obtener opciones válidas según el tipo de encoder
            if hasattr(encoder, 'classes_'):
                # Es un LabelEncoder
                valid_options = list(encoder.classes_)
            elif isinstance(encoder, dict):
                # Es un diccionario de mapeo
                valid_options = list(encoder.keys())
            else:
                print(f" Advertencia: Encoder desconocido para '{col}'. Saltando...")
                continue
            
            print(f"\n🔹 Dato: {col.upper()}")
            print(f"   Opciones válidas: {', '.join(valid_options)}")
            
            while True:
                val = input(f"     Escribe una opción: ").strip()
                
                # Validación: puede ser case-insensitive para los diccionarios
                if isinstance(encoder, dict):
                    # Para diccionarios, aceptamos cualquier capitalización
                    if val.title() in valid_options or val in valid_options:
                        user_data[col] = val
                        break
                else:
                    # Para LabelEncoder, debe coincidir exactamente
                    if val in valid_options:
                        user_data[col] = val
                        break
                
                print(f"     Valor incorrecto. Copia exactamente una de las opciones de arriba.")

        # --- CASO B: Columna NUMÉRICA (Edad, Dinero, etc) ---
        else:
            print(f"\n🔹 Dato: {col.upper()}")
            while True:
                val = input(f"     Introduce un número: ").strip()
                try:
                    # Intentamos convertir a número
                    float_val = float(val)
                    user_data[col] = float_val
                    break
                except ValueError:
                    print("     Eso no es un número válido. Inténtalo de nuevo.")
    
    return user_data
def main():
    # 1. Comprobar si hay que entrenar
    if not check_is_trained():
        print(">>> Modelo no encontrado. Iniciando entrenamiento...")
        try:
            # IMPORTANTE: Asegúrate de que 'credit-train.csv' (con columna 'y') está en data/raw/
            df = load_data("customer_churn_dataset-training-master.csv") 
            
            # Preprocesamos y guardamos los artefactos (encoders)
            X_train, X_test, y_train, y_test = preprocess_data(df, target_col='Churn', save_artifacts=True)
            
            # Entrenamos
            models = train_models(X_train, y_train)
            evaluate_models(models, X_test, y_test)
            print(">>> Entrenamiento finalizado.")
        except Exception as e:
            print(f" Error fatal durante el entrenamiento: {e}")
            return
    else:
        print(">>> Modelo cargado correctamente.")

    # 2. Cargar el modelo ya entrenado
    try:
        model = joblib.load(MODELS_DIR / MODEL_NAME)
    except FileNotFoundError:
        print(f" No se pudo cargar el modelo {MODEL_NAME}.")
        return

    # 3. Bucle infinito para pedir datos
    while True:
        try:
            # Pedir datos (ahora con validación robusta)
            raw_data = ask_user_data()
            if not raw_data: break 

            # Procesar (convertir texto a números y escalar)
            processed_data = process_input(raw_data)
            
            # Predecir
            prediction = model.predict(processed_data)[0]
            
            # Intentar sacar probabilidad si el modelo lo soporta
            probs = model.predict_proba(processed_data)[0] if hasattr(model, "predict_proba") else [0,0]
            prob_yes = probs[1] if len(probs) > 1 else 0

            # Mostrar resultado
            print("\n" + "-"*30)
            if prediction == 1:
                print(f" RIESGO DE CHURN ALTO (Probabilidad: {prob_yes:.1%})")
            else:
                print(f" Cliente estable (Riesgo bajo - Confianza NO churn: {probs[0]:.1%})")
            print("-"*30 + "\n")

            
            # ¿Otra vez?
            if input("¿Evaluar otro cliente? (s/n): ").lower() != 's':
                print("Cerrando programa...")
                break
                
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f" Ocurrió un error inesperado: {e}")
            break

if __name__ == "__main__":
    main()