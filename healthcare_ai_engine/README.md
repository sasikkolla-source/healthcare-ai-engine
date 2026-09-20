# Healthcare AI Engine (Starter Base)

A simple, self-contained scaffold for a healthcare risk-prediction system:
a scikit-learn model behind a Flask API, with a small web UI to try it out.

> **Disclaimer:** This is an educational/portfolio starter project. It trains
> on **synthetic** data and is **not** a validated medical device. Do not use
> it for real diagnostic, treatment, or clinical decisions.

## What's inside

```
healthcare_ai_engine/
├── app.py                  # Flask app: web UI + /api/predict, /api/health
├── train_model.py          # Trains and saves the risk-prediction model
├── data/
│   └── generate_data.py    # Synthetic patient dataset generator
├── models/                 # Trained model is saved here (risk_model.joblib)
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css           # Web UI styling
└── requirements.txt
```

## How it works

1. `data/generate_data.py` creates a synthetic dataset of patients (age, BMI,
   blood pressure, glucose, cholesterol, smoking status, exercise habits,
   family history) with a risk label derived from a simplified, made-up
   scoring rule plus noise.
2. `train_model.py` trains two candidate models (logistic regression and a
   random forest) inside a scikit-learn `Pipeline` (with feature scaling),
   evaluates both on a held-out test set, and saves the better-performing one
   to `models/risk_model.joblib`.
3. `app.py` loads that saved model and serves:
   - `GET /` — a form-based UI for entering patient data and viewing the
     predicted risk band (Low / Moderate / High) and probability.
   - `POST /api/predict` — JSON API. Send the 8 feature fields, get back a
     prediction, probability, and risk band.
   - `GET /api/health` — basic health check.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates data + trains + saves the model)
python train_model.py

# 4. Run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

## API example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "bmi": 29.4,
    "systolic_bp": 138,
    "glucose": 112,
    "cholesterol": 210,
    "smoker": 1,
    "exercise_hours_per_week": 1.0,
    "family_history": 1
  }'
```

Response:

```json
{
  "prediction": 1,
  "risk_probability": 0.71,
  "risk_band": "High",
  "model_used": "random_forest",
  "disclaimer": "Educational demo only. Not a medical device. Not for clinical or diagnostic use."
}
```

## Extending this base

This is intentionally a minimal foundation. Natural next steps:

- **Real data**: swap `data/generate_data.py` for a real, properly licensed
  and de-identified clinical dataset (with a clinician involved in feature
  and label design), and re-run `train_model.py`.
- **More conditions**: train separate models (or a multi-label model) for
  different risk conditions instead of one generic "disease_risk" label.
- **Explainability**: add SHAP or permutation-importance output so each
  prediction comes with a "why" (e.g., top contributing factors).
- **Persistence**: add a database to store patient records and prediction
  history instead of stateless single requests.
- **Auth & compliance**: if this will ever touch real patient data, add
  authentication, audit logging, encryption at rest/in transit, and review
  HIPAA (or your local equivalent) requirements before going further.
- **Testing**: add unit tests for `validate_input`, `risk_band`, and the API
  routes (e.g., with `pytest` + Flask's test client).
