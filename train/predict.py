"""Saved model se naye applicant ka default risk predict karo.
Run: python train/predict.py
"""
from pathlib import Path
import joblib
import pandas as pd

model = joblib.load(Path(__file__).resolve().parent.parent / "models" / "loan_model.pkl")

applicant = pd.DataFrame([{
    "age": 28, "income": 42000, "loan_amount": 35000, "credit_score": 590,
    "employment_years": 1.5, "existing_loans": 3,
    "home_ownership": "Rent", "loan_purpose": "Personal",
}])
applicant["loan_to_income"] = applicant["loan_amount"] / applicant["income"]

p = model.predict_proba(applicant)[0, 1]
print(f"Default probability: {p:.1%} -> {'HIGH RISK' if p >= 0.5 else 'LOW RISK'}")
