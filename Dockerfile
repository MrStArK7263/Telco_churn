# Dockerfile  —  Customer Churn Prediction API (FastAPI + uvicorn)
# Deployed on Render as a Web Service using this Dockerfile.
#
# Build locally:
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 -e API_KEY=your_key churn-api
#
# Render sets the PORT environment variable automatically.
# The API_KEY must be added in Render's environment settings — never in this file.

FROM python:3.11-slim

# System dependencies required by XGBoost (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer — cached unless requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only the files the backend needs at runtime
COPY api.py          .
COPY churn_features.py .
COPY models/         models/

# Render injects PORT at runtime; default to 8000 for local docker run
ENV PORT=8000

# Start the uvicorn server bound to all interfaces on Render's PORT.
# Module name:  api   (the file is api.py)
# ASGI object:  app   (the FastAPI() instance in api.py)
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
