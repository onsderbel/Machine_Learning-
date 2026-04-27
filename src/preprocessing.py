"""
preprocessing.py — Full preprocessing pipeline.

Steps:
 1. Load raw data
 2. Drop useless features (constant, ID)
 3. Parse RegistrationDate → extract RegYear, RegMonth, RegDay, RegWeekday
 4. Engineer features from LastLoginIP
 5. Fix aberrant values (SupportTicketsCount, SatisfactionScore)
 6. Impute missing values
 7. Encode categorical features
 8. Remove highly correlated features
 9. Normalize numerical features (StandardScaler)
10. Save processed data & train/test splits
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent dir to path so utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_raw_data, DATA_PROC, DATA_TT, REPORTS_DIR,
    basic_eda, plot_missing_values, plot_churn_distribution,
    plot_correlation_heatmap, get_features_to_drop_by_correlation
)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Drop useless / constant columns
# ─────────────────────────────────────────────────────────────────────────────
COLS_TO_DROP = [
    "NewsletterSubscribed",   # 100% constant ("Yes")
    "LastLoginIP",            # raw IP — we'll engineer features before dropping
    "RegistrationDate",       # raw text — we'll parse before dropping
]


# ─────────────────────────────────────────────────────────────────────────────
#  2. Parse RegistrationDate
# ─────────────────────────────────────────────────────────────────────────────
def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], dayfirst=True, errors="coerce"
    )
    df["RegYear"]    = df["RegistrationDate"].dt.year
    df["RegMonth"]   = df["RegistrationDate"].dt.month
    df["RegDay"]     = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday
    print(f"[parse] RegistrationDate → 4 new features. NaT count: {df['RegistrationDate'].isna().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3. Engineer features from LastLoginIP
# ─────────────────────────────────────────────────────────────────────────────
PRIVATE_PREFIXES = ("10.", "192.168.", "172.")


def is_private_ip(ip: str) -> int:
    if pd.isna(ip) or not isinstance(ip, str):
        return -1
    return int(any(ip.startswith(p) for p in PRIVATE_PREFIXES))


def extract_ip_first_octet(ip: str) -> float:
    if pd.isna(ip) or not isinstance(ip, str):
        return np.nan
    parts = ip.split(".")
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return np.nan


def engineer_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IP_IsPrivate"]   = df["LastLoginIP"].apply(is_private_ip)
    df["IP_FirstOctet"]  = df["LastLoginIP"].apply(extract_ip_first_octet)
    print("[engineer] IP features added: IP_IsPrivate, IP_FirstOctet")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  4. Fix aberrant values
# ─────────────────────────────────────────────────────────────────────────────
def fix_aberrant_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    SupportTicketsCount : -1 and 999 are sentinel values → NaN
    SatisfactionScore   : -1 and 99 are sentinel values → NaN
    """
    df = df.copy()
    df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)
    df["SatisfactionScore"]   = df["SatisfactionScore"].replace([-1, 99], np.nan)
    print("[fix] Aberrant values in SupportTicketsCount & SatisfactionScore → NaN")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  5. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MonetaryPerDay"]   = df["MonetaryTotal"] / (df["Recency"] + 1)
    df["AvgBasketValue"]   = df["MonetaryTotal"] / df["Frequency"].replace(0, np.nan)
    df["TenureRatio"]      = df["Recency"] / (df["CustomerTenureDays"] + 1)
    df["FreqPerTenure"]    = df["Frequency"] / (df["CustomerTenureDays"] + 1)
    print("[engineer] New features: MonetaryPerDay, AvgBasketValue, TenureRatio, FreqPerTenure")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  6. Impute missing values
# ─────────────────────────────────────────────────────────────────────────────
def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Age (30% missing): KNN imputation
    - Other numeric NaN (AvgDaysBetweenPurchases, SupportTicketsCount,
      SatisfactionScore, IP_FirstOctet): median imputation
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # KNN for Age (most meaningful imputation for demographic feature)
    if "Age" in num_cols and df["Age"].isna().sum() > 0:
        knn_imp = KNNImputer(n_neighbors=5)
        age_cols = [c for c in num_cols if c != "Churn"]
        df[age_cols] = knn_imp.fit_transform(df[age_cols])
        print("[impute] Age → KNN imputation")

    # Median for remaining numeric NaN
    remaining_nan = [c for c in num_cols if df[c].isna().sum() > 0]
    if remaining_nan:
        med_imp = SimpleImputer(strategy="median")
        df[remaining_nan] = med_imp.fit_transform(df[remaining_nan])
        print(f"[impute] Median imputation: {remaining_nan}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  7. Encode categorical features
# ─────────────────────────────────────────────────────────────────────────────

# Ordinal feature mappings (domain-ordered)
ORDINAL_MAPS = {
    "RFMSegment":          {"Dormants": 0, "Potentiels": 1, "Fidèles": 2, "Champions": 3},
    "AgeCategory":         {"Inconnu": -1, "18-24": 0, "25-34": 1, "35-44": 2,
                             "45-54": 3, "55-64": 4, "65+": 5},
    "SpendingCategory":    {"Low": 0, "Medium": 1, "High": 2, "VIP": 3},
    "LoyaltyLevel":        {"Inconnu": -1, "Nouveau": 0, "Jeune": 1,
                             "Établi": 2, "Ancien": 3},
    "ChurnRiskCategory":   {"Faible": 0, "Moyen": 1, "Élevé": 2, "Critique": 3},
    "BasketSizeCategory":  {"Inconnu": -1, "Petit": 0, "Moyen": 1, "Grand": 2},
    "PreferredTimeOfDay":  {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4},
}

# One-hot encoded features
ONE_HOT_COLS = [
    "CustomerType",
    "FavoriteSeason",
    "Region",
    "WeekendPreference",
    "ProductDiversity",
    "Gender",
    "AccountStatus",
]

# Target encode Country (high cardinality 37+)
TARGET_ENCODE_COL = "Country"


def encode_categoricals(df: pd.DataFrame,
                         target_means: dict = None,
                         fit: bool = True) -> tuple:
    """
    Returns (df_encoded, target_means_dict).
    target_means is passed during transform (test set) to avoid leakage.
    """
    df = df.copy()

    # Ordinal
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            print(f"[encode] Ordinal: {col}")

    # One-hot
    existing_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=False, dtype=int)
    print(f"[encode] One-Hot: {existing_ohe}")

    # Target encode Country
    if TARGET_ENCODE_COL in df.columns:
        if fit:
            # Use mean of Churn per country (computed on training set)
            target_means = df.groupby(TARGET_ENCODE_COL)["Churn"].mean().to_dict()
        global_mean = df["Churn"].mean()
        df[TARGET_ENCODE_COL + "_enc"] = (
            df[TARGET_ENCODE_COL].map(target_means).fillna(global_mean)
        )
        df.drop(columns=[TARGET_ENCODE_COL], inplace=True)
        print(f"[encode] Target encode: {TARGET_ENCODE_COL}")

    return df, target_means


# ─────────────────────────────────────────────────────────────────────────────
#  8. Drop correlated features
# ─────────────────────────────────────────────────────────────────────────────
def drop_correlated(df: pd.DataFrame, threshold: float = 0.85) -> tuple:
    """Returns (df_reduced, dropped_cols)."""
    to_drop = get_features_to_drop_by_correlation(df, threshold)
    to_drop = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[corr] Dropped {len(to_drop)} correlated features: {to_drop}")
    return df, to_drop


# ─────────────────────────────────────────────────────────────────────────────
#  9. Scale numerical features
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Fit scaler on train, apply to both. Returns (X_train_sc, X_test_sc, scaler)."""
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_sc = X_train.copy()
    X_test_sc  = X_test.copy()

    X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[scale] StandardScaler applied to {len(num_cols)} numeric columns.")
    return X_train_sc, X_test_sc, scaler, num_cols


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing():
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(DATA_TT,   exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE — START")
    print("="*60)

    # ── Load ────────────────────────────────────────────────────────────────
    df = load_raw_data()
    basic_eda(df)

    # ── EDA plots ───────────────────────────────────────────────────────────
    plot_missing_values(df)
    plot_churn_distribution(df)

    # ── Step 1: Parse RegistrationDate ──────────────────────────────────────
    df = parse_registration_date(df)

    # ── Step 2: Engineer IP features ────────────────────────────────────────
    df = engineer_ip_features(df)

    # ── Step 3: Drop raw/constant columns ───────────────────────────────────
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    df.drop(columns=["RegistrationDate"], inplace=True, errors="ignore")
    print(f"[drop] Dropped: {COLS_TO_DROP + ['RegistrationDate']}")

    # ── Step 4: Fix aberrant values ─────────────────────────────────────────
    df = fix_aberrant_values(df)

    # ── Step 5: Feature engineering ─────────────────────────────────────────
    df = feature_engineering(df)

    # ── Step 6: Impute missing values ────────────────────────────────────────
    df = impute_numeric(df)

    # ── Step 7: Encode categoricals ─────────────────────────────────────────
    df, target_means = encode_categoricals(df, fit=True)

    # ── Step 8: Drop CustomerID (not a feature) ─────────────────────────────
    df.drop(columns=["CustomerID"], inplace=True, errors="ignore")

    # ── Step 9: Drop highly correlated features ──────────────────────────────
    plot_correlation_heatmap(df)
    df, dropped_cols = drop_correlated(df, threshold=0.85)

    print(f"\n[shape] After full preprocessing: {df.shape}")

    # ── Save cleaned dataset ─────────────────────────────────────────────────
    df.to_csv(os.path.join(DATA_PROC, "customers_clean.csv"), index=False)
    print(f"[save] Cleaned dataset → data/processed/customers_clean.csv")

    # ── Step 10: Train/test split ────────────────────────────────────────────
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[split] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[split] Churn ratio — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # ── Step 11: Scale AFTER split (no data leakage) ─────────────────────────
    X_train_sc, X_test_sc, scaler, num_cols = scale_features(X_train, X_test)

    # ── Save splits ──────────────────────────────────────────────────────────
    X_train_sc.to_csv(os.path.join(DATA_TT, "X_train.csv"), index=False)
    X_test_sc.to_csv(os.path.join(DATA_TT,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_TT,    "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_TT,     "y_test.csv"),  index=False)
    print("[save] Train/test splits → data/train_test/")

    # ── Save scaler metadata ─────────────────────────────────────────────────
    import joblib
    os.makedirs(os.path.join(os.path.dirname(DATA_PROC), "..", "models"), exist_ok=True)
    from utils import MODELS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(num_cols,     os.path.join(MODELS_DIR, "num_cols.joblib"))
    joblib.dump(target_means, os.path.join(MODELS_DIR, "target_means.joblib"))
    joblib.dump(dropped_cols, os.path.join(MODELS_DIR, "dropped_cols.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_names.joblib"))
    print("[save] Scaler & metadata → models/")

    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE — DONE ✓")
    print("="*60 + "\n")

    return X_train_sc, X_test_sc, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
