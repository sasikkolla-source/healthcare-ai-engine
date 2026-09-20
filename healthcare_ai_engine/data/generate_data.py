"""
generate_data.py
-----------------
Generates a synthetic patient dataset for training the Healthcare AI Engine's
risk-prediction model.

NOTE: This is SYNTHETIC data created for demonstration purposes only. It is
NOT real patient data and the underlying "rules" are simplified approximations
of known risk factors (age, BMI, blood pressure, glucose, smoking, exercise,
family history). For a production system, replace this with a real, properly
licensed and de-identified clinical dataset, and involve a clinician in
feature/label design.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_patient_data(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, n_samples)
    bmi = rng.normal(27, 5, n_samples).clip(15, 55)
    systolic_bp = rng.normal(125, 18, n_samples).clip(80, 220)
    glucose = rng.normal(100, 25, n_samples).clip(60, 300)
    cholesterol = rng.normal(190, 35, n_samples).clip(100, 350)
    smoker = rng.binomial(1, 0.2, n_samples)
    exercise_hours_per_week = rng.exponential(2.5, n_samples).clip(0, 20)
    family_history = rng.binomial(1, 0.25, n_samples)

    # Composite risk score built from known-direction risk factors, plus noise.
    # This is a simplified heuristic, not a validated clinical formula.
    # Each term is expressed directly in "logit units" so the noise term below
    # is calibrated relative to them (keeps the label learnable but not trivial).
    risk_score = (
        0.028 * (age - 40)
        + 0.07 * (bmi - 25)
        + 0.022 * (systolic_bp - 120)
        + 0.028 * (glucose - 100)
        + 0.012 * (cholesterol - 180)
        + 0.9 * smoker
        - 0.10 * exercise_hours_per_week
        + 0.8 * family_history
        + rng.normal(0, 0.6, n_samples)  # noise: adds realistic irreducible error
    )

    probability = 1 / (1 + np.exp(-risk_score))
    disease_risk = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": np.round(bmi, 1),
            "systolic_bp": np.round(systolic_bp, 0),
            "glucose": np.round(glucose, 0),
            "cholesterol": np.round(cholesterol, 0),
            "smoker": smoker,
            "exercise_hours_per_week": np.round(exercise_hours_per_week, 1),
            "family_history": family_history,
            "disease_risk": disease_risk,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_patient_data()
    out_path = "data/patient_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic patient records -> {out_path}")
    print(df["disease_risk"].value_counts(normalize=True))
