"""
utils.py — Reusable helper functions for the ML pipeline.
Covers: data loading, EDA, correlation analysis, plotting, metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, mean_squared_error, r2_score
)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
DATA_TT     = os.path.join(BASE_DIR, "data", "train_test")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_CSV = os.path.join(DATA_RAW, "retail_customers_COMPLETE_CATEGORICAL.csv")


# ─────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """Load original dataset from data/raw/."""
    df = pd.read_csv(RAW_CSV)
    print(f"[load] {df.shape[0]} rows × {df.shape[1]} columns loaded.")
    return df


def load_processed_data() -> pd.DataFrame:
    path = os.path.join(DATA_PROC, "customers_clean.csv")
    df = pd.read_csv(path)
    print(f"[load] Processed data: {df.shape}")
    return df


# ─────────────────────────────────────────────
#  EDA helpers
# ─────────────────────────────────────────────
def basic_eda(df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing values, and basic stats."""
    print("=" * 60)
    print(f"Shape      : {df.shape}")
    print(f"Duplicates : {df.duplicated().sum()}")
    print("\n--- Data types ---")
    print(df.dtypes.value_counts())
    print("\n--- Missing values (%) ---")
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(missing[missing > 0].round(2))
    print("\n--- Numeric stats ---")
    print(df.describe().T.round(2))
    print("=" * 60)


def plot_missing_values(df: pd.DataFrame, save: bool = True) -> None:
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("[missing] No missing values found.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Missing Values (%)")
    ax.set_ylabel("%")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "missing_values.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()


def plot_churn_distribution(df: pd.DataFrame, save: bool = True) -> None:
    counts = df["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_xticklabels(["Loyal (0)", "Churned (1)"], rotation=0)
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "churn_distribution.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
#  Correlation & multicollinearity
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.8,
                              save: bool = True) -> None:
    """Plot correlation heatmap of numeric features."""
    num = df.select_dtypes(include=[np.number]).drop(
        columns=["CustomerID", "Churn"], errors="ignore"
    )
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()


def get_highly_correlated_pairs(df: pd.DataFrame,
                                 threshold: float = 0.8) -> pd.DataFrame:
    """Return pairs of features with |correlation| > threshold."""
    num = df.select_dtypes(include=[np.number]).drop(
        columns=["CustomerID", "Churn"], errors="ignore"
    )
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Feature_A", "level_1": "Feature_B", 0: "Correlation"})
        .query(f"Correlation >= {threshold}")
        .sort_values("Correlation", ascending=False)
    )
    return pairs


def get_features_to_drop_by_correlation(df: pd.DataFrame,
                                         threshold: float = 0.8) -> list:
    """Return list of redundant features to drop (keep one from each pair)."""
    pairs = get_highly_correlated_pairs(df, threshold)
    to_drop = set()
    for _, row in pairs.iterrows():
        if row["Feature_A"] not in to_drop:
            to_drop.add(row["Feature_B"])
    return list(to_drop)


# ─────────────────────────────────────────────
#  PCA helpers
# ─────────────────────────────────────────────
def run_pca_analysis(X_scaled: np.ndarray, n_components: int = 10,
                     save: bool = True) -> PCA:
    """Fit PCA and plot explained variance."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, n_components + 1), explained * 100, color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("PCA — Scree Plot")

    axes[1].plot(range(1, n_components + 1), cumulative * 100, marker="o", color="tomato")
    axes[1].axhline(85, linestyle="--", color="gray", label="85% threshold")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("PCA — Cumulative Variance")
    axes[1].legend()

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "pca_variance.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()

    # Print summary
    for i, (exp, cum) in enumerate(zip(explained, cumulative), 1):
        print(f"  PC{i}: {exp*100:.1f}% variance  |  cumulative: {cum*100:.1f}%")

    return pca


# ─────────────────────────────────────────────
#  Classification metrics
# ─────────────────────────────────────────────
def evaluate_classifier(model, X_test, y_test, model_name: str = "Model",
                         save: bool = True) -> dict:
    """Print classification metrics and save confusion matrix plot."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Loyal", "Churned"]))

    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC : {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Loyal", "Churned"],
                yticklabels=["Loyal", "Churned"], ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        path = os.path.join(REPORTS_DIR, fname)
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()

    # ROC curve
    if y_prob is not None and save:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue", lw=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} — ROC Curve")
        ax.legend()
        plt.tight_layout()
        fname = model_name.lower().replace(" ", "_") + "_roc_curve.png"
        path = os.path.join(REPORTS_DIR, fname)
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
        plt.close()

    return {"model": model_name, "auc": auc, "report": classification_report(y_test, y_pred)}


# ─────────────────────────────────────────────
#  Regression metrics
# ─────────────────────────────────────────────
def evaluate_regressor(model, X_test, y_test, model_name: str = "Regressor",
                        save: bool = True) -> dict:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {model_name} — Regression Metrics")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.3, color="steelblue", s=10)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_") + "_actual_vs_predicted.png"
        path = os.path.join(REPORTS_DIR, fname)
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()

    return {"model": model_name, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────
#  Feature importance plot
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             model_name: str = "Model", save: bool = True) -> None:
    if not hasattr(model, "feature_importances_"):
        print("[feature_importance] Model has no feature_importances_ attribute.")
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_") + "_feature_importance.png"
        path = os.path.join(REPORTS_DIR, fname)
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved: {path}")
    plt.close()
