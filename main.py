import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

from tasa_churn.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from tasa_churn.data.make_dataset import load_data
from tasa_churn.features.build_features import preprocess_data, process_input
from tasa_churn.models.train_model import train_models
from tasa_churn.models.predict_model import evaluate_models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BEST_MODEL_RECORD = MODELS_DIR / "best_model.txt"
TRAINING_FILE = "customer_churn_dataset-training-master.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_best_model_name() -> str:
    """Returns the filename of the best model saved during training."""
    if BEST_MODEL_RECORD.exists():
        name = BEST_MODEL_RECORD.read_text().strip()
        if name:
            return name
    # Fallback: pick any .joblib in MODELS_DIR that isn't an artifact
    candidates = [
        p.name for p in MODELS_DIR.glob("*.joblib")
        if "artifact" not in p.name.lower()
    ]
    if candidates:
        return candidates[0]
    return "RandomForest.joblib"


def is_trained() -> bool:
    """Returns True when a trained model and its artifacts are present."""
    model_name = get_best_model_name()
    model_ok = (MODELS_DIR / model_name).exists()
    artifacts_ok = (ARTIFACTS_DIR / "encoders.joblib").exists()
    return model_ok and artifacts_ok


def load_artifacts() -> tuple:
    """
    Loads encoders, scaler config and column order from disk once.
    Returns (columns, encoders).
    Raises SystemExit on missing files so the caller does not need to handle it.
    """
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
        return columns, encoders
    except FileNotFoundError as exc:
        logger.error("Artifact not found: %s", exc)
        logger.error("Delete the 'models/' folder and re-run to retrain.")
        sys.exit(1)


def ask_user_data(columns: list, encoders: dict) -> dict | None:
    """
    Prompts the user for each feature interactively with strict validation.
    Returns a dict of {column: value}, or None if the user aborts with Ctrl-C.
    """
    print("\n" + "=" * 42)
    print("   CHURN RISK PREDICTION")
    print("=" * 42)

    user_data: dict = {}

    for col in columns:
        # --- Categorical column ---
        if col in encoders:
            encoder = encoders[col]

            if hasattr(encoder, "classes_"):
                valid_options = list(encoder.classes_)
            elif isinstance(encoder, dict):
                valid_options = list(encoder.keys())
            else:
                logger.warning("Unknown encoder type for column '%s'. Skipping.", col)
                continue

            print(f"\n  {col.upper()}")
            print(f"  Options: {', '.join(valid_options)}")

            while True:
                val = input("  > ").strip()
                if isinstance(encoder, dict):
                    if val.title() in valid_options or val in valid_options:
                        user_data[col] = val
                        break
                else:
                    if val in valid_options:
                        user_data[col] = val
                        break
                print("  Invalid value. Choose one of the options above.")

        # --- Numeric column ---
        else:
            print(f"\n  {col.upper()}")
            while True:
                val = input("  > ").strip()
                try:
                    user_data[col] = float(val)
                    break
                except ValueError:
                    print("  Not a valid number. Try again.")

    return user_data


def display_result(prediction: int, probs: list) -> None:
    """Prints the churn prediction result in a consistent format."""
    prob_churn = probs[1] if len(probs) > 1 else 0.0
    prob_stable = probs[0] if len(probs) > 0 else 0.0

    print("\n" + "-" * 42)
    if prediction == 1:
        print(f"  HIGH CHURN RISK  (probability: {prob_churn:.1%})")
    else:
        print(f"  LOW RISK — stable client  (confidence: {prob_stable:.1%})")
    print("-" * 42 + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training() -> None:
    """Loads data, preprocesses, trains models and saves the best one."""
    logger.info("No trained model found. Starting training pipeline...")
    try:
        df = load_data(TRAINING_FILE)
        X_train, X_test, y_train, y_test = preprocess_data(
            df, target_col="Churn", save_artifacts=True
        )
        models = train_models(X_train, y_train)
        best_name = evaluate_models(models, X_test, y_test)

        # Persist the name of the winner so main always loads the right file
        if best_name:
            BEST_MODEL_RECORD.write_text(best_name)
            logger.info("Best model: %s", best_name)

        logger.info("Training complete.")
    except Exception:
        logger.exception("Fatal error during training.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not is_trained():
        run_training()
    else:
        logger.info("Trained model found. Skipping training.")

    model_name = get_best_model_name()
    model_path = MODELS_DIR / model_name

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded: %s", model_name)
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    # Load artifacts once, reuse across all predictions in the session
    columns, encoders = load_artifacts()

    while True:
        try:
            raw_data = ask_user_data(columns, encoders)
            if not raw_data:
                break

            processed = process_input(raw_data)
            prediction = model.predict(processed)[0]
            probs = (
                model.predict_proba(processed)[0]
                if hasattr(model, "predict_proba")
                else [0.0, 0.0]
            )

            display_result(prediction, list(probs))

            answer = input("Evaluate another client? (y/n): ").strip().lower()
            if answer != "y":
                logger.info("Session closed.")
                break

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except ValueError as exc:
            # Validation errors should not kill the session
            logger.warning("Input error: %s — please try again.", exc)
        except Exception:
            logger.exception("Unexpected error.")
            break


if __name__ == "__main__":
    main()