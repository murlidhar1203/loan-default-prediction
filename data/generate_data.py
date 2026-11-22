"""Synthetic loan dataset generate karta hai (real bank data nahi hai).
Run: python data/generate_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

age = rng.integers(21, 66, N)
income = rng.normal(55000, 18000, N).clip(15000, 180000).round(-2)
loan_amount = (income * rng.uniform(0.1, 1.2, N)).round(-2)
credit_score = rng.normal(680, 70, N).clip(300, 850).round()
employment_years = np.minimum(rng.exponential(5, N), age - 20).round(1)
home_ownership = rng.choice(["Rent", "Mortgage", "Own"], N, p=[0.45, 0.35, 0.20])
loan_purpose = rng.choice(["Education", "Medical", "Home", "Business", "Personal"], N)
existing_loans = rng.poisson(1.2, N)

# Default probability: realistic-ish rules + noise
dti = loan_amount / income
z = (-3.0 + 2.5 * dti - 0.012 * (credit_score - 650)
     - 0.08 * employment_years + 0.25 * existing_loans
     + 0.5 * (home_ownership == "Rent") - 0.3 * (home_ownership == "Own")
     + rng.normal(0, 0.8, N))
# non-linear effect (RF ko edge milta hai): high credit-risk + high DTI combo
z = z + 1.2 * ((dti > 0.8) & (credit_score < 620))
prob = 1 / (1 + np.exp(-z))
default = np.where(rng.random(N) < prob, "Yes", "No")

df = pd.DataFrame({
    "customer_id": [f"L{i:04d}" for i in range(1, N + 1)],
    "age": age, "income": income, "loan_amount": loan_amount,
    "credit_score": credit_score, "employment_years": employment_years,
    "home_ownership": home_ownership, "loan_purpose": loan_purpose,
    "existing_loans": existing_loans, "default": default,
})
# thoda missing data, taaki imputation practise ho
for col in ["income", "employment_years"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

out = Path(__file__).resolve().parent / "loan_data.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows -> {out}  | default rate: {(df.default=='Yes').mean():.1%}")
