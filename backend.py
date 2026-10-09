"""
backend.py  -  Model loading and prediction logic for the Customer Churn Predictor.

Responsibilities:
  * Load the saved model pipeline and metadata once at startup.
  * Accept a raw customer dictionary (as collected from the Streamlit widgets) and
    return a structured prediction result.
  * All file paths are resolved relative to this file so the app works both
    locally and on Streamlit Community Cloud.

This module does NOT import streamlit and has NO UI code, making it independently
testable and reusable (e.g. from api.py or a CLI script).
"""
import json
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

import churn_features  # noqa: F401  <- joblib needs this importable to deserialise the pipeline

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file, not the current working directory
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent
_MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_PATH = _BASE_DIR / "models" / f"churn_model_v{_MODEL_VERSION}.joblib"
META_PATH = _BASE_DIR / "models" / f"churn_model_v{_MODEL_VERSION}_metadata.json"
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """
    Load and return the saved sklearn pipeline from disk.

    Raises FileNotFoundError if the model file is missing.
    Raises any joblib / pickle error if the file is corrupt or the
    library versions are incompatible.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure models/churn_model_v{version}.joblib is committed to the repo."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    """
    Load and return the model metadata dict from disk.
    Returns an empty dict if the metadata file is missing (non-fatal).
    """
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(model, customer_data: dict) -> tuple:
    """
    Run the model on a single customer dictionary.

    Parameters
    ----------
    model        : the loaded sklearn pipeline (returned by load_model())
    customer_data: dict whose keys match the raw training column names, as
                   assembled by app.py from the Streamlit widgets.

    Returns
    -------
    (probability, error_message)
        Exactly one of the two values is None.
        - On success: (float in [0, 1], None)
        - On failure: (None, human-readable error string)
    """
    try:
        row = pd.DataFrame([customer_data])          # one-row DataFrame, same columns as training
        probability = float(model.predict_proba(row)[0, 1])   # P(class=1) == P(churn)
        return probability, None
    except Exception as exc:
        return None, f"Prediction failed: {exc}"


def classify(probability: float) -> dict:
    """
    Turn a raw churn probability into a structured result dict.

    Parameters
    ----------
    probability : float in [0, 1]

    Returns
    -------
    dict with keys:
        churn_probability  - the raw float
        will_churn         - bool  (True when probability >= THRESHOLD)
        prediction_label   - "WILL CHURN" | "WILL STAY"
        risk_level         - "High" | "Medium" | "Low"
        threshold          - the decision threshold used
    """
    will_churn = probability >= THRESHOLD
    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": probability,
        "will_churn": will_churn,
        "prediction_label": "WILL CHURN" if will_churn else "WILL STAY",
        "risk_level": risk,
        "threshold": THRESHOLD,
    }
