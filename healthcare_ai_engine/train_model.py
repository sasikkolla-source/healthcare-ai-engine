"""
train_model.py
----------------
Trains the Healthcare AI Engine's risk-prediction model on the synthetic
patient dataset and saves the fitted pipeline (scaler + classifier) to disk.

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.generate_data import generate_patient_data

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "glucose",
    "cholesterol",
    "smoker",
    "exercise_hours_per_week",
    "family_history",
]
TARGET_COLUMN = "disease_risk"
MODEL_PATH = "models/risk_model.joblib"


def load_dataset() -> pd.DataFrame:
    csv_path = "data/patient_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Generate on the fly if no CSV has been saved yet
    return generate_patient_data()


def train() -> None:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Two candidate models; we pick the one with the better ROC-AUC on the
    # held-out test set. Swap/extend this list to experiment with others.
    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    best_name, best_pipeline, best_auc = None, None, -1.0

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}  |  ROC-AUC: {auc:.3f}")
        print(classification_report(y_test, preds, digits=3))

        if auc > best_auc:
            best_name, best_pipeline, best_auc = name, pipeline, auc

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": best_name,
        },
        MODEL_PATH,
    )
    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.3f}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
