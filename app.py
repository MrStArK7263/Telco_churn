"""
app.py  -  Customer Churn Predictor  (Streamlit front end)

Run locally:
    streamlit run app.py

Deployment:
    Push to GitHub and connect to Streamlit Community Cloud.
    Select app.py as the entrypoint.
    Set API_URL and API_KEY in the app's Secrets settings
    (or in .streamlit/secrets.toml for local use — never commit that file).

Modes:
    1. Render API (recommended for production):
         Set API_URL to your Render service URL.
         Set API_KEY to the same secret configured in Render's environment.
         The app sends X-API-Key in every request.
    2. Local model file (fallback when API_URL is not set):
         The app loads backend.py directly — no API server needed.
         Useful for quick local testing without Docker.
"""
import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Configuration — read from Streamlit secrets first, then env vars.
#    Neither value is ever hardcoded here.
# ---------------------------------------------------------------------------
def _secret(key: str, default: str = "") -> str:
    """Read a value from st.secrets (Cloud) or os.environ (local / CI)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


MODEL_VERSION = _secret("MODEL_VERSION", os.getenv("MODEL_VERSION", "1.0.0"))
API_URL = _secret("API_URL", "").rstrip("/")   # empty = use local backend.py
API_KEY = _secret("API_KEY", "")              # sent as X-API-Key header to Render API

# Build the auth header once; used in every outbound request when a key is set.
_AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# ---------------------------------------------------------------------------
# 2. Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)
st.title("📉 Customer Churn Predictor")
st.write(
    "Enter the customer details and click **Predict** to see whether "
    "this customer is likely to leave."
)


# ---------------------------------------------------------------------------
# 3. Load model metadata (and optionally the local model)
#    When API_URL is set the local model is never loaded — the Render API
#    handles everything. The @st.cache_resource wrapper means this runs only
#    once per session regardless of how many times the user clicks Predict.
# ---------------------------------------------------------------------------
# backend.py and churn_features are only needed in local-model mode.
# Import them lazily so the Streamlit Cloud app doesn't need them when
# API_URL is set (they're still listed in requirements.txt for the fallback).
import backend          # noqa: E402
import churn_features   # noqa: E402, F401  (joblib needs this on the path)


@st.cache_resource
def get_model_and_metadata():
    """Load the local sklearn pipeline + metadata. Called only in local mode."""
    model = backend.load_model()
    metadata = backend.load_metadata()
    return model, metadata


model, metadata = None, {}
if not API_URL:
    # ---- local-model mode ----
    try:
        model, metadata = get_model_and_metadata()
    except FileNotFoundError as err:
        st.error(str(err))
        st.info(
            "Commit `models/churn_model_v1.0.0.joblib` and "
            "`models/churn_model_v1.0.0_metadata.json` to your repository."
        )
        st.stop()
    except Exception as err:
        st.error(f"Could not load the model file `churn_model_v{MODEL_VERSION}.joblib`.")
        st.exception(err)
        st.info(
            "Most common cause: the scikit-learn / pandas / xgboost versions in "
            "requirements.txt differ from those used to train the model."
        )
        st.stop()
else:
    # ---- Render API mode — fetch metadata for the sidebar ----
    try:
        resp = requests.get(
            f"{API_URL}/model-info",
            headers=_AUTH_HEADERS,
            timeout=8,
        )
        if resp.status_code == 200:
            metadata = resp.json()
        elif resp.status_code == 401:
            st.warning("API returned 401 Unauthorized. Check that API_KEY matches the Render setting.")
        else:
            st.warning(f"Could not fetch model info from API (HTTP {resp.status_code}).")
    except requests.exceptions.ConnectionError:
        st.warning(
            "Could not reach the Render API. "
            "Check that API_URL is correct and the service is running."
        )
    except requests.exceptions.Timeout:
        st.warning("Request to /model-info timed out. The service may still be starting up.")
    except Exception:
        pass   # non-fatal — sidebar will show 'unknown'


# ---------------------------------------------------------------------------
# 4. Sidebar: model information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model information")
    st.write(f"**Mode:** {'Render API' if API_URL else 'Local model file'}")
    st.write(f"**Model:** {metadata.get('model_name', 'unknown')}")
    st.write(f"**Version:** {metadata.get('model_version', MODEL_VERSION)}")
    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        st.write("**Test-set scores**")
        st.table(pd.DataFrame(test_metrics, index=["score"]).T)


# ---------------------------------------------------------------------------
# 5. Input form (three columns)
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.number_input(
        "Tenure (months with the company)", min_value=0, max_value=120, value=12
    )

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":                          # business rule: no phone -> no multiple lines
        multiple = "No phone service"
        st.selectbox("Multiple lines", ["No phone service"], disabled=True)
    else:
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])

    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addon_names = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    addons = {}
    for name in addon_names:
        if internet == "No":                   # business rule: no internet -> no add-ons
            addons[name] = "No internet service"
        else:
            addons[name] = st.selectbox(name, ["No", "Yes"])
    if internet == "No":
        st.info("No internet → all internet add-ons are set to 'No internet service'.")

with col3:
    st.subheader("Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input(
        "Monthly charges", min_value=1.0, max_value=500.0, value=70.0, step=0.5
    )
    total = st.number_input(
        "Total charges (0 = estimate automatically)", min_value=0.0, value=0.0, step=10.0
    )

# Build the raw customer dict. Keys must match the training column names.
customer = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    **addons,                                  # six add-on columns
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total) if total > 0 else None,   # None -> estimated in churn_features.py
}


# ---------------------------------------------------------------------------
# 6. Prediction dispatcher
# ---------------------------------------------------------------------------
def run_prediction(customer_data: dict) -> tuple:
    """
    Returns (result_dict | None, error_message | None).

    API mode:   POST to Render with X-API-Key header; parse PredictionOutput.
    Local mode: call backend.predict() + backend.classify() directly.
    """
    if API_URL:
        # --- Render API mode ---
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                headers=_AUTH_HEADERS,
                timeout=15,
            )
        except requests.exceptions.ConnectionError:
            return None, (
                "Cannot reach the Render API. "
                "Check that API_URL is correct and the service is running."
            )
        except requests.exceptions.Timeout:
            return None, "The API request timed out. The service may be starting up — try again."

        if response.status_code == 200:
            data = response.json()
            result = {
                "churn_probability": data["churn_probability"],
                "will_churn": data["prediction"] == "Yes",
                "prediction_label": "WILL CHURN" if data["prediction"] == "Yes" else "WILL STAY",
                "risk_level": data["risk_level"],
                "threshold": data["threshold"],
            }
            return result, None

        if response.status_code == 401:
            return None, "API returned 401 Unauthorized. Check that API_KEY matches the Render setting."

        if response.status_code == 422:
            msgs = "; ".join(
                e.get("msg", "") for e in response.json().get("detail", [])
            )
            return None, f"Invalid input: {msgs}"

        if response.status_code == 503:
            return None, "The API model is not loaded yet. Check /health on the Render service."

        return None, f"Unexpected API error (HTTP {response.status_code})."

    # --- Local model mode ---
    probability, error = backend.predict(model, customer_data)
    if error:
        return None, error
    return backend.classify(probability), None


# ---------------------------------------------------------------------------
# 7. Predict button and results
# ---------------------------------------------------------------------------
if st.button("🔮 Predict", type="primary"):
    result, error_message = run_prediction(customer)

    if error_message:
        st.error(error_message)
    else:
        left, right = st.columns(2)
        with left:
            if result["will_churn"]:
                st.error(f"### ⚠️ Predicted class: {result['prediction_label']}")
            else:
                st.success(f"### ✅ Predicted class: {result['prediction_label']}")
            st.metric("Churn probability", f"{result['churn_probability']:.1%}")
            st.progress(min(max(result["churn_probability"], 0.0), 1.0))
            st.write(
                f"**Risk level:** {result['risk_level']}   |   "
                f"decision threshold = {result['threshold']}"
            )
        with right:
            st.write("**Most important features in the model (global explanation)**")
            top_features = pd.DataFrame(metadata.get("top_features", []))
            if top_features.empty:
                st.caption("Feature importance not available.")
            else:
                st.bar_chart(top_features.set_index("feature")["importance"])
