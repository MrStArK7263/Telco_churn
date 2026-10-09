"""
churn_features.py
-----------------
Shared cleaning + feature-engineering code.

The SAME function is used in 3 places so training and prediction never disagree:
  1. the Jupyter notebook (training)
  2. the FastAPI service (api.py)
  3. the saved model pipeline (model.joblib)
"""
import numpy as np
import pandas as pd

# Columns that only make sense when the customer has internet service
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

# Final column lists used by the model pipeline (after feature engineering)
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                "avg_monthly_spend", "charge_ratio", "num_addon_services",
                "monthly_per_service", "has_family", "is_autopay",
                "streaming_user", "has_security_support", "fiber_monthly"]

CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService"] + ADDON_COLS + \
                   ["Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"]

MODEL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def clean_data(df):
    """Row-wise cleaning that does NOT learn anything from the data (no leakage)."""
    df = df.copy()

    # 1. Remove stray spaces from every text column ("Yes " -> "Yes")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("object").str.strip()

    # 2. TotalCharges is text because some cells are blank (" ") -> make it a number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # 3. Customers with tenure 0 are brand new -> they have not been billed yet
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # 4. Otherwise TotalCharges is roughly tenure x MonthlyCharges
    can_estimate = df["TotalCharges"].isna() & df["tenure"].notna() & df["MonthlyCharges"].notna()
    df.loc[can_estimate, "TotalCharges"] = df.loc[can_estimate, "tenure"] * df.loc[can_estimate, "MonthlyCharges"]

    # 5. Fix missing values using business logic
    #    No internet  -> every add-on must be "No internet service"
    no_net = df["InternetService"] == "No"
    for col in ADDON_COLS:
        df.loc[no_net & df[col].isna(), col] = "No internet service"
    #    No phone -> MultipleLines must be "No phone service" (and the reverse)
    df.loc[(df["PhoneService"] == "No") & df["MultipleLines"].isna(), "MultipleLines"] = "No phone service"
    df.loc[df["PhoneService"].isna() & (df["MultipleLines"] == "No phone service"), "PhoneService"] = "No"
    df.loc[df["PhoneService"].isna() & df["MultipleLines"].isin(["Yes", "No"]), "PhoneService"] = "Yes"
    return df


def add_features(df):
    """Create new, meaningful columns from the existing ones."""
    df = df.copy()
    tenure_safe = df["tenure"].clip(lower=1)          # avoid dividing by zero

    # --- ratios ---
    df["avg_monthly_spend"] = df["TotalCharges"] / tenure_safe
    df["charge_ratio"] = np.where(df["avg_monthly_spend"] > 0,
                                  df["MonthlyCharges"] / df["avg_monthly_spend"], 1.0)

    # --- combining columns ---
    df["num_addon_services"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["monthly_per_service"] = df["MonthlyCharges"] / (df["num_addon_services"] + 1)
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["is_autopay"] = df["PaymentMethod"].fillna("").str.contains("automatic").astype(int)
    df["streaming_user"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["has_security_support"] = ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)

    # --- interaction feature (fibre + month-to-month is a classic churn combo) ---
    df["fiber_monthly"] = ((df["InternetService"] == "Fiber optic") &
                           (df["Contract"] == "Month-to-month")).astype(int)

    # --- binning a number into groups ---
    bins = [-1, 12, 24, 48, 100]
    labels = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype("object")
    return df


def build_features(df):
    """clean_data + add_features + keep only the model columns.
    This is the function stored inside the saved pipeline."""
    df = add_features(clean_data(df))
    return df[MODEL_INPUT_COLS]
