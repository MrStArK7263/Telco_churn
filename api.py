"""
api.py  -  FastAPI service that serves the churn model.

Run locally:   uvicorn api:app --reload --port 8000
Docs (auto):   http://localhost:8000/docs

Settings come from environment variables (see .env.example):
    MODEL_VERSION  -> which model file to load  (default 1.0.0)
    MODEL_DIR      -> override the models folder (default: models/ next to this file)
    THRESHOLD      -> probability above which we say "will churn" (default 0.5)
    LOG_LEVEL      -> INFO / DEBUG / WARNING
    API_KEY        -> when set, every request must include header X-API-Key with this value.
                      Leave unset to disable auth (useful for local dev and tests).
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, model_validator

import churn_features  # noqa: F401  (needed so joblib can find build_features when loading)

# ----------------------------------------------------------------------------
# 1. Settings from environment variables
# ----------------------------------------------------------------------------
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_API_KEY = os.getenv("API_KEY", "")   # empty string = auth disabled

# Resolve model paths relative to this file so they work inside the Docker
# container regardless of the working directory uvicorn starts in.
_BASE_DIR = Path(__file__).parent
_model_dir_override = os.getenv("MODEL_DIR", "")
MODEL_DIR = Path(_model_dir_override) if _model_dir_override else _BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / f"churn_model_v{MODEL_VERSION}.joblib"
META_PATH = MODEL_DIR / f"churn_model_v{MODEL_VERSION}_metadata.json"

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("churn-api")

# The model lives in this dictionary after start-up
state = {"model": None, "metadata": {}}


# ----------------------------------------------------------------------------
# 2. API-key authentication
#    - When API_KEY env-var is set: every request must send X-API-Key header.
#    - When API_KEY is not set:     auth is skipped (local dev / CI tests).
# ----------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str = Security(_api_key_header)):
    """Dependency injected into every protected endpoint."""
    if not _API_KEY:
        return   # auth disabled — allow all requests
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it in the X-API-Key header.",
        )


# ----------------------------------------------------------------------------
# 3. Load the model once when the server starts
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            state["metadata"] = json.loads(META_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded model %s from %s", MODEL_VERSION, MODEL_PATH)
    except Exception as exc:  # keep the server alive so /health can report the problem
        logger.exception("Could not load model: %s", exc)
    yield


app = FastAPI(title="Customer Churn Prediction API", version=MODEL_VERSION, lifespan=lifespan)


# ----------------------------------------------------------------------------
# 4. Input validation (Pydantic does this automatically -> HTTP 422 if wrong)
# ----------------------------------------------------------------------------
YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., gt=0, le=500)
    TotalCharges: Optional[float] = Field(None, ge=0, description="Leave empty to estimate it")

    @model_validator(mode="after")
    def check_business_rules(self):
        add_ons = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection,
                   self.TechSupport, self.StreamingTV, self.StreamingMovies]
        if self.InternetService == "No" and any(a != "No internet service" for a in add_ons):
            raise ValueError("When InternetService is 'No', every add-on must be 'No internet service'")
        if self.InternetService != "No" and any(a == "No internet service" for a in add_ons):
            raise ValueError("'No internet service' is only allowed when InternetService is 'No'")
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("When PhoneService is 'No', MultipleLines must be 'No phone service'")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'")
        return self


class PredictionOutput(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    threshold: float
    model_version: str


# ----------------------------------------------------------------------------
# 5. Error handling
# ----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again later."})


def model_not_ready():
    return JSONResponse(status_code=503, content={"detail": "Model is not loaded yet. Check /health."})


# ----------------------------------------------------------------------------
# 6. Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check — used by Render and load balancers. No auth required."""
    loaded = state["model"] is not None
    body = {"status": "ok" if loaded else "unhealthy", "model_loaded": loaded, "model_version": MODEL_VERSION}
    return JSONResponse(status_code=200 if loaded else 503, content=body)


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    """Model version, test metrics and the most important features."""
    if state["model"] is None:
        return model_not_ready()
    return {"model_version": MODEL_VERSION, **state["metadata"]}


@app.post("/predict", response_model=PredictionOutput, dependencies=[Depends(require_api_key)])
def predict(customer: CustomerInput):
    if state["model"] is None:
        return model_not_ready()

    start = time.perf_counter()
    row = pd.DataFrame([customer.model_dump()])          # 1 row, same columns as training
    probability = float(state["model"].predict_proba(row)[0, 1])
    label = "Yes" if probability >= THRESHOLD else "No"
    risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
    logger.info("prediction=%s prob=%.3f time=%.1fms", label, probability, (time.perf_counter() - start) * 1000)

    return PredictionOutput(prediction=label, churn_probability=round(probability, 4),
                            risk_level=risk, threshold=THRESHOLD, model_version=MODEL_VERSION)
