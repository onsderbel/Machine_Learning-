"""
train_model.py — Model training: Clustering, Classification (Churn), Regression.

Models trained:
  • K-Means clustering (customer segmentation)
  • Logistic Regression (churn classification — baseline)
  • Random Forest Classifier (churn classification — main)
  • Random Forest Regressor (MonetaryTotal prediction)
  • PCA (dimensionality reduction, used for clustering viz)

All models saved to models/ as .joblib files.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import silhouette_score
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[warn] imbalanced-learn not installed. Using class_weight='balanced' instead of SMOTE.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    MODELS_DIR, DATA_TT, REPORTS_DIR,
    evaluate_classifier, evaluate_regressor,
    plot_feature_importance, run_pca_analysis
)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Load train/test splits
# ─────────────────────────────────────────────────────────────────────────────
def load_splits():
    X_train = pd.read_csv(os.path.join(DATA_TT, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_TT, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_TT, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_TT, "y_test.csv")).squeeze()
    print(f"[load] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[load] Churn distribution in train: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
#  1. PCA — Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────
def train_pca(X_train: pd.DataFrame, n_components: int = 10) -> PCA:
    print("\n" + "─"*50)
    print("  PCA — Dimensionality Reduction")
    print("─"*50)
    pca = run_pca_analysis(X_train.values, n_components=n_components)

    # Choose components explaining 85% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.argmax(cumvar >= 0.85)) + 1
    print(f"[pca] {n_keep} components explain ≥85% variance.")

    # Refit with optimal n_components
    pca_final = PCA(n_components=n_keep, random_state=42)
    pca_final.fit(X_train.values)
    joblib.dump(pca_final, os.path.join(MODELS_DIR, "pca.joblib"))
    print(f"[save] PCA model → models/pca.joblib")
    return pca_final


# ─────────────────────────────────────────────────────────────────────────────
#  2. K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────
def train_kmeans(X_train: pd.DataFrame, pca: PCA,
                 k_range: range = range(2, 9)) -> KMeans:
    print("\n" + "─"*50)
    print("  K-Means Clustering")
    print("─"*50)

    X_pca = pca.transform(X_train.values)

    # Elbow + Silhouette
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, labels)
        silhouettes.append(sil)
        print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(list(k_range), silhouettes, "o-", color="tomato")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "kmeans_elbow_silhouette.png"), dpi=150)
    plt.close()

    # Best k by silhouette
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\n[kmeans] Best k = {best_k} (highest silhouette: {max(silhouettes):.4f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km_final.fit(X_pca)

    # Visualize clusters in 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train.values)
    labels = km_final.predict(X_pca)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                          alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"K-Means Clusters (k={best_k}) — PCA 2D projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "kmeans_clusters_2d.png"), dpi=150)
    plt.close()

    joblib.dump(km_final, os.path.join(MODELS_DIR, "kmeans.joblib"))
    print(f"[save] KMeans → models/kmeans.joblib")
    return km_final


# ─────────────────────────────────────────────────────────────────────────────
#  3. Churn Classification — Logistic Regression (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def train_logistic(X_train, X_test, y_train, y_test):
    print("\n" + "─"*50)
    print("  Logistic Regression — Churn Baseline")
    print("─"*50)

    # Handle class imbalance with class_weight
    lr = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced", C=1.0
    )
    lr.fit(X_train, y_train)

    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"[cv] 5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    evaluate_classifier(lr, X_test, y_test, model_name="Logistic Regression")

    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    print(f"[save] Logistic Regression → models/logistic_regression.joblib")
    return lr


# ─────────────────────────────────────────────────────────────────────────────
#  4. Churn Classification — Random Forest (main model)
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest_classifier(X_train, X_test, y_train, y_test):
    print("\n" + "─"*50)
    print("  Random Forest Classifier — Churn Prediction")
    print("─"*50)

    # Balance classes via SMOTE (if available) or class_weight
    if HAS_SMOTE:
        print("[smote] Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"[smote] After resampling: {pd.Series(y_res).value_counts().to_dict()}")
    else:
        print("[balance] Using class_weight='balanced' (install imbalanced-learn for SMOTE).")
        X_res, y_res = X_train, y_train

    # Hyperparameter search
    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight":      ["balanced"],
    }
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(
        rf, param_grid, cv=5, scoring="roc_auc",
        n_jobs=-1, verbose=1
    )
    gs.fit(X_res, y_res)

    best_rf = gs.best_estimator_
    print(f"\n[gs] Best params : {gs.best_params_}")
    print(f"[gs] Best CV AUC : {gs.best_score_:.4f}")

    evaluate_classifier(best_rf, X_test, y_test,
                         model_name="Random Forest Classifier")
    plot_feature_importance(best_rf, list(X_train.columns),
                             model_name="Random Forest Classifier")

    joblib.dump(best_rf, os.path.join(MODELS_DIR, "rf_classifier.joblib"))
    print(f"[save] RF Classifier → models/rf_classifier.joblib")
    return best_rf


# ─────────────────────────────────────────────────────────────────────────────
#  5. Regression — MonetaryTotal prediction
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest_regressor(X_train, X_test, y_train_raw, y_test_raw):
    """
    Predict MonetaryTotal from the other features.
    We load raw processed data to get the original MonetaryTotal column
    (before it's the y target — here we use all features except Churn).
    """
    print("\n" + "─"*50)
    print("  Random Forest Regressor — MonetaryTotal Prediction")
    print("─"*50)

    from utils import DATA_PROC
    df_clean = pd.read_csv(os.path.join(DATA_PROC, "customers_clean.csv"))

    from sklearn.model_selection import train_test_split as tts
    y_reg = df_clean["MonetaryTotal"]
    X_reg = df_clean.drop(columns=["MonetaryTotal", "Churn"])

    # Re-encode: only numeric subset to avoid re-encoding issues
    X_reg = X_reg.select_dtypes(include=[np.number])

    X_tr, X_te, y_tr, y_te = tts(X_reg, y_reg,
                                   test_size=0.2, random_state=42)

    rfr = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        random_state=42, n_jobs=-1
    )
    rfr.fit(X_tr, y_tr)

    evaluate_regressor(rfr, X_te, y_te, model_name="Random Forest Regressor")
    plot_feature_importance(rfr, list(X_tr.columns),
                             model_name="Random Forest Regressor")

    joblib.dump(rfr, os.path.join(MODELS_DIR, "rf_regressor.joblib"))
    print(f"[save] RF Regressor → models/rf_regressor.joblib")
    return rfr


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_training():
    print("\n" + "="*60)
    print("  TRAINING PIPELINE — START")
    print("="*60)

    X_train, X_test, y_train, y_test = load_splits()

    pca   = train_pca(X_train, n_components=10)
    km    = train_kmeans(X_train, pca)
    lr    = train_logistic(X_train, X_test, y_train, y_test)
    rf_cl = train_random_forest_classifier(X_train, X_test, y_train, y_test)
    rf_rg = train_random_forest_regressor(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("  TRAINING PIPELINE — DONE ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_training()
