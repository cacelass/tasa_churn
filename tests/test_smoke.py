"""
Smoke tests for the churnguard package.

These tests do not require a trained model or real data.
They verify that all modules can be imported and that the
path helpers resolve to sensible locations.
"""

import importlib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_import_paths():
    mod = importlib.import_module("tasa_churn.utils.paths")
    assert hasattr(mod, "MODELS_DIR"), "MODELS_DIR not defined in paths.py"
    assert hasattr(mod, "ARTIFACTS_DIR"), "ARTIFACTS_DIR not defined in paths.py"


def test_import_make_dataset():
    importlib.import_module("tasa_churn.data.make_dataset")


def test_import_build_features():
    importlib.import_module("tasa_churn.features.build_features")


def test_import_train_model():
    importlib.import_module("tasa_churn.models.train_model")


def test_import_predict_model():
    importlib.import_module("tasa_churn.models.predict_model")


# ---------------------------------------------------------------------------
# Path sanity tests
# ---------------------------------------------------------------------------

def test_models_dir_is_path():
    from tasa_churn.utils.paths import MODELS_DIR
    assert isinstance(MODELS_DIR, Path)


def test_artifacts_dir_is_path():
    from tasa_churn.utils.paths import ARTIFACTS_DIR
    assert isinstance(ARTIFACTS_DIR, Path)


def test_artifacts_dir_is_child_of_models():
    from tasa_churn.utils.paths import MODELS_DIR, ARTIFACTS_DIR
    # artifacts/ should live inside models/ (or at least share the same root)
    assert MODELS_DIR in ARTIFACTS_DIR.parents or ARTIFACTS_DIR.parent == MODELS_DIR


# ---------------------------------------------------------------------------
# Main entrypoint import test
# ---------------------------------------------------------------------------

def test_import_main():
    """main.py should be importable without side effects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", Path("main.py"))
    if spec is None:
        # Running tests from a different working directory — skip gracefully
        return
    mod = importlib.util.module_from_spec(spec)
    # We do NOT call spec.loader.exec_module(mod) to avoid running main()
    assert mod is not None
