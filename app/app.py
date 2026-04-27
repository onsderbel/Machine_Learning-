"""
app.py — Flask web application for churn prediction.

Run: python app/app.py
Visit: http://127.0.0.1:5000
"""

import os
import sys

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flask import Flask, request, render_template, jsonify
from predict import predict_churn, load_artifacts

app = Flask(__name__)

# Load models once at startup
print("[app] Loading models...")
ARTIFACTS = load_artifacts()
print("[app] Models loaded.")


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body or form data with customer features.
    Returns prediction as JSON.
    """
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
        # Convert numeric strings to float
        for key, val in data.items():
            try:
                data[key] = float(val)
            except (ValueError, TypeError):
                pass  # leave as string

    result = predict_churn(data, ARTIFACTS)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": bool(ARTIFACTS)})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
