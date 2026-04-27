"""
predict.py — Prediction interface for new customer data.

Usage (CLI):
    python src/predict.py

Or import predict_churn() in Flask app.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS_DIR, DATA_PROC
from preprocessing import (
    parse_registration_date, engineer_ip_features,
    fix_aberrant_values, feature_engineering,
    encode_categoricals, ORDINAL_MAPS, ONE_HOT_COLS
)


# ─────────────────────────────────────────────────────────────────────────────
#  Load saved artifacts
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    artifacts = {}
    files = {
        "rf_classifier":  "rf_classifier.joblib",
        "scaler":         "scaler.joblib",
        "num_cols":       "num_cols.joblib",
        "target_means":   "target_means.joblib",
        "dropped_cols":   "dropped_cols.joblib",
        "feature_names":  "feature_names.joblib",
        "pca":            "pca.joblib",
        "kmeans":         "kmeans.joblib",
    }
    for key, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            print(f"[warn] {path} not found — run preprocessing.py and train_model.py first.")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocess a single new customer dict
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_new_customer(customer_dict: dict, artifacts: dict) -> pd.DataFrame:
    """
    Takes a dict of raw customer fields and returns a scaled DataFrame
    ready for prediction.
    """
    df = pd.DataFrame([customer_dict])

    # Repeat preprocessing steps
    df = parse_registration_date(df)
    df = engineer_ip_features(df)

    cols_to_drop = ["NewsletterSubscribed", "LastLoginIP", "RegistrationDate"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    df = fix_aberrant_values(df)
    df = feature_engineering(df)

    # Impute numeric NaN with column means from training (use 0 as fallback)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0)

    # Encode categoricals (use saved target_means, fit=False)
    df["Churn"] = 0  # dummy target for encoding
    df, _ = encode_categoricals(df, target_means=artifacts.get("target_means"), fit=False)
    df.drop(columns=["Churn", "CustomerID"], inplace=True, errors="ignore")

    # Drop correlated columns removed during training
    dropped = artifacts.get("dropped_cols", [])
    df.drop(columns=[c for c in dropped if c in df.columns], inplace=True)

    # Align columns to training features (add missing as 0, drop extras)
    feature_names = artifacts.get("feature_names", [])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale using saved scaler
    scaler   = artifacts.get("scaler")
    nc       = artifacts.get("num_cols", [])
    valid_nc = [c for c in nc if c in df.columns]
    if scaler and valid_nc:
        df[valid_nc] = scaler.transform(df[valid_nc])

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Predict churn for a single customer
# ─────────────────────────────────────────────────────────────────────────────
def predict_churn(customer_dict: dict, artifacts: dict = None) -> dict:
    """
    Returns:
        {
          "churn_prediction": 0 or 1,
          "churn_probability": float,
          "risk_label": "Low" | "Medium" | "High" | "Critical",
          "cluster": int
        }
    """
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts.get("rf_classifier")
    if model is None:
        return {"error": "Model not found. Run train_model.py first."}

    X = preprocess_new_customer(customer_dict, artifacts)

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    # Risk label
    if prob < 0.25:
        risk = "Low"
    elif prob < 0.50:
        risk = "Medium"
    elif prob < 0.75:
        risk = "High"
    else:
        risk = "Critical"

    # Cluster assignment
    cluster = -1
    pca_model = artifacts.get("pca")
    km_model  = artifacts.get("kmeans")
    if pca_model and km_model:
        X_pca = pca_model.transform(X.values)
        cluster = int(km_model.predict(X_pca)[0])

    return {
        "churn_prediction":  pred,
        "churn_probability": round(prob, 4),
        "risk_label":        risk,
        "cluster":           cluster,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch prediction on CSV
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(input_csv: str, output_csv: str, artifacts: dict = None):
    if artifacts is None:
        artifacts = load_artifacts()

    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        r = predict_churn(row.to_dict(), artifacts)
        results.append(r)

    out = df.copy()
    out["PredictedChurn"]      = [r.get("churn_prediction",  -1) for r in results]
    out["ChurnProbability"]    = [r.get("churn_probability",  -1) for r in results]
    out["RiskLabel"]           = [r.get("risk_label",   "N/A") for r in results]
    out["AssignedCluster"]     = [r.get("cluster",           -1) for r in results]
    out.to_csv(output_csv, index=False)
    print(f"[predict] Batch predictions saved to {output_csv}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example customer
    example = {
        "CustomerID":              12345,
        "Recency":                 45,
        "Frequency":               8,
        "MonetaryTotal":           1500.0,
        "MonetaryAvg":             187.5,
        "MonetaryStd":             50.0,
        "MonetaryMin":             50.0,
        "MonetaryMax":             400.0,
        "TotalQuantity":           320,
        "AvgQuantityPerTransaction": 40.0,
        "MinQuantity":             -5,
        "MaxQuantity":             200,
        "CustomerTenureDays":      400,
        "FirstPurchaseDaysAgo":    420,
        "PreferredDayOfWeek":      2,
        "PreferredHour":           14,
        "PreferredMonth":          11,
        "WeekendPurchaseRatio":    0.3,
        "AvgDaysBetweenPurchases": 35.0,
        "UniqueProducts":          45,
        "UniqueDescriptions":      43,
        "AvgProductsPerTransaction": 5.6,
        "UniqueCountries":         1,
        "NegativeQuantityCount":   2,
        "ZeroPriceCount":          0,
        "CancelledTransactions":   1,
        "ReturnRatio":             0.05,
        "TotalTransactions":       350,
        "UniqueInvoices":          8,
        "AvgLinesPerInvoice":      43.0,
        "Age":                     34.0,
        "RegistrationDate":        "2010-05-15",
        "NewsletterSubscribed":    "Yes",
        "LastLoginIP":             "77.99.123.45",
        "SupportTicketsCount":     2.0,
        "SatisfactionScore":       4.0,
        "RFMSegment":              "Fidèles",
        "AgeCategory":             "25-34",
        "SpendingCategory":        "Medium",
        "CustomerType":            "Régulier",
        "FavoriteSeason":          "Automne",
        "PreferredTimeOfDay":      "Après-midi",
        "Region":                  "UK",
        "LoyaltyLevel":            "Établi",
        "ChurnRiskCategory":       "Faible",
        "WeekendPreference":       "Semaine",
        "BasketSizeCategory":      "Moyen",
        "ProductDiversity":        "Modéré",
        "Gender":                  "M",
        "AccountStatus":           "Active",
        "Country":                 "United Kingdom",
    }

    print("\n[predict] Running prediction for example customer...")
    artifacts = load_artifacts()
    result = predict_churn(example, artifacts)
    print("\n--- Prediction Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
