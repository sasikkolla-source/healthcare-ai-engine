"""
app.py
------
Flask application exposing the Healthcare AI Engine:
  - GET  /                 -> simple web UI for entering patient data
  - POST /api/predict      -> JSON API for risk prediction
  - GET  /api/health       -> health check

IMPORTANT DISCLAIMER
This project is an educational / portfolio starter template. It is NOT a
medical device, has NOT been clinically validated, and must NOT be used to
make real diagnostic or treatment decisions. Always consult a qualified
healthcare professional.
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "models/risk_model.joblib"

app = Flask(__name__)

_model_bundle = None  # lazy-loaded


def get_model_bundle():
    """Load the trained model pipeline from disk (cached after first call)."""
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                "Run `python train_model.py` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def validate_input(payload: dict) -> tuple[dict, list[str]]:
    """Validate and coerce incoming patient data. Returns (clean_data, errors)."""
    schema = {
        "age": (0, 120),
        "bmi": (10, 70),
        "systolic_bp": (60, 260),
        "glucose": (40, 400),
        "cholesterol": (80, 500),
        "smoker": (0, 1),
        "exercise_hours_per_week": (0, 40),
        "family_history": (0, 1),
    }

    clean = {}
    errors = []

    for field, (low, high) in schema.items():
        if field not in payload:
            errors.append(f"Missing field: {field}")
            continue
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            errors.append(f"Field '{field}' must be numeric.")
            continue
        if not (low <= value <= high):
            errors.append(f"Field '{field}' must be between {low} and {high}.")
            continue
        clean[field] = value

    return clean, errors


def risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    model_available = os.path.exists(MODEL_PATH)
    return jsonify({"status": "ok", "model_available": model_available})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    clean_data, errors = validate_input(payload)
    if errors:
        return jsonify({"error": "Invalid input.", "details": errors}), 400

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    X = pd.DataFrame([clean_data])[feature_columns]
    probability = float(pipeline.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)

    return jsonify(
        {
            "prediction": prediction,
            "risk_probability": round(probability, 4),
            "risk_band": risk_band(probability),
            "model_used": bundle.get("model_name", "unknown"),
            "disclaimer": (
                "Educational demo only. Not a medical device. "
                "Not for clinical or diagnostic use."
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
