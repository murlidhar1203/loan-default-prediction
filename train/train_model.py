"""Loan default prediction: train, evaluate, save model.
Run: python train/train_model.py   (kisi bhi folder se chalega)
"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "loan_data.csv"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

NUM = ["age", "income", "loan_amount", "credit_score", "employment_years",
       "existing_loans", "loan_to_income"]
CAT = ["home_ownership", "loan_purpose"]

df = pd.read_csv(DATA)
df["loan_to_income"] = df["loan_amount"] / df["income"]   # feature engineering
X = df[NUM + CAT]
y = df["default"].map({"Yes": 1, "No": 0})

# stratify => train/test dono mein same default ratio
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def make_pipe(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])

# Do models compare karte hain, winner 5-fold CV ROC-AUC se chunte hain
# (test set se nahi, warna data leakage hota hai)
lr = GridSearchCV(
    make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
    {"model__C": [0.1, 1, 10]}, cv=5, scoring="roc_auc", n_jobs=-1)
rf = GridSearchCV(
    make_pipe(RandomForestClassifier(random_state=42, class_weight="balanced")),
    {"model__n_estimators": [100, 200], "model__max_depth": [4, 6, 10],
     "model__min_samples_leaf": [5, 20]},
    cv=5, scoring="roc_auc", n_jobs=-1)
lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)
print(f"CV ROC-AUC  LogisticRegression: {lr.best_score_:.3f} | RandomForest: {rf.best_score_:.3f}")
winner = rf if rf.best_score_ >= lr.best_score_ else lr
best, name = winner.best_estimator_, type(winner.best_estimator_.named_steps["model"]).__name__
print(f"Selected: {name} {winner.best_params_}")

proba = best.predict_proba(X_te)[:, 1]
pred = best.predict(X_te)
print(f"{name} test ROC-AUC: {roc_auc_score(y_te, proba):.3f}\n")
report = classification_report(y_te, pred, target_names=["No default", "Default"])
print(report)
(REPORTS / "metrics.txt").write_text(
    f"Model: {name} {winner.best_params_}\nTest ROC-AUC: {roc_auc_score(y_te, proba):.3f}\n\n{report}")

# Plots
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["No", "Yes"])
plt.title("Confusion Matrix"); plt.savefig(REPORTS / "confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close()

names = best.named_steps["pre"].get_feature_names_out()
m = best.named_steps["model"]
vals = m.feature_importances_ if hasattr(m, "feature_importances_") else abs(m.coef_[0])
pd.Series(vals, index=names).sort_values().tail(10).plot.barh(title=f"Top 10 Features ({name})")
plt.savefig(REPORTS / "feature_importance.png", dpi=120, bbox_inches="tight"); plt.close()

joblib.dump(best, MODELS / "loan_model.pkl")
print(f"Model saved -> {MODELS / 'loan_model.pkl'}")
