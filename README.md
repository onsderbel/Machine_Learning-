# Analyse Comportementale Clientèle Retail — ML Project

> E-commerce gift shop customer behavior analysis using the full ML pipeline:
> Exploration → Preprocessing → Modeling → Evaluation → Deployment

---

## Project Structure

```
projet_ml_retail/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned & encoded data
│   └── train_test/             # Train/test splits
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
├── src/
│   ├── utils.py                # Helpers & reusable functions
│   ├── preprocessing.py        # Full preprocessing pipeline
│   ├── train_model.py          # Training: clustering, classification, regression
│   └── predict.py              # Prediction for new customers
├── models/                     # Saved models (.joblib)
├── app/
│   ├── app.py                  # Flask web application
│   └── templates/
│       └── index.html          # Web UI
├── reports/                    # Generated plots & reports
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd projet_ml_retail

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocessing
```bash
python src/preprocessing.py
```
Outputs cleaned data to `data/processed/` and train/test splits to `data/train_test/`.

### Step 2 — Train Models
```bash
python src/train_model.py
```
Trains clustering (K-Means), classification (Churn prediction), and regression (Monetary prediction). Models saved to `models/`.

### Step 3 — Predict
```bash
python src/predict.py
```

### Step 4 — Launch Flask App
```bash
python app/app.py
```
Then open your browser at: `http://127.0.0.1:5000`

---

## Dataset

- **4372 customers**, **52 features**
- Covers RFM metrics, behavioral data, demographics, account info
- Target variable: `Churn` (0 = loyal, 1 = churned)

---

## Models Used

| Task | Algorithm |
|---|---|
| Clustering | K-Means |
| Classification (Churn) | Random Forest + Logistic Regression |
| Regression (Monetary) | Random Forest Regressor |
| Dimensionality Reduction | PCA |

---

## Author

GI2 — Machine Learning Workshop 2025-2026
