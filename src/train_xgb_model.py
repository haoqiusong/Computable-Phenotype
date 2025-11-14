"""
XGBoost Model Training, Calibration, Evaluation, and Feature Importance
======================================================================

This script:
1. Loads the processed dataset.
2. Performs preprocessing, filtering, and scaling.
3. Trains an XGBoost classifier with internal class weighting.
4. Calibrates predicted probabilities (Platt scaling).
5. Evaluates performance using AUROC, AUPRC, MCC, sensitivity, specificity, etc.
6. Generates calibration plots.
7. Saves performance metrics, plots, and feature importance rankings.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)


# =====================================================================
# 1) Load Dataset
# =====================================================================

df = pd.read_csv("dataset.csv")

# ---------------------------------------------------------------------
# Basic filtering steps (customize for your dataset)
# ---------------------------------------------------------------------
# Example: remove rows with invalid target label
df = df[df["Label"] != -1]

# Store basic metadata (replace with your own column names)
patient_ids = df["Patient_ID"]
epoch_times = df["Epoch_Time"]
target = df["Label"]

# =====================================================================
# 2) Column Removal (Replaced with generic placeholders)
# =====================================================================

# Replace these placeholders with your own dataset-specific drop lists
cols_to_drop = ["Var1", "Var2", "Var3"]

df = df.drop(columns=cols_to_drop, errors="ignore")

# Final feature matrix
X = df
y = target.reset_index(drop=True)

# =====================================================================
# 3) Train / Test Split
# =====================================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================================
# 4) Scaling
# =====================================================================

scaler = MinMaxScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# =====================================================================
# 5) Calibration Split
# =====================================================================

X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, stratify=y_train_full, random_state=42
)

# =====================================================================
# 6) Train XGBoost Model
# =====================================================================

# Compute class weight ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = neg / pos if pos > 0 else 1

model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    learning_rate=0.01,
    n_estimators=500,
    max_depth=10,
    max_delta_step=1.5
)

model.fit(X_train, y_train)

# =====================================================================
# 7) Probability Calibration
# =====================================================================

calibrated_model = CalibratedClassifierCV(
    estimator=model, method="sigmoid", cv="prefit"
)
calibrated_model.fit(X_calib, y_calib)

# Prediction threshold (adjust for your use case)
threshold = 0.1
y_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# =====================================================================
# 8) Evaluation Metrics
# =====================================================================

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

metrics = {
    "AUROC": roc_auc_score(y_test, y_prob),
    "AUPRC": average_precision_score(y_test, y_prob),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Sensitivity": recall_score(y_test, y_pred),
    "Specificity": tn / (tn + fp),
    "F1": f1_score(y_test, y_pred),
    "PPV": precision_score(y_test, y_pred),
    "NPV": tn / (tn + fn)
}

print("\n===== Calibrated Model Performance =====")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# =====================================================================
# 9) ROC Curve
# =====================================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# =====================================================================
# 10) Precision–Recall Curve
# =====================================================================

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 7))
plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.close()

# =====================================================================
# 11) Feature Importance Plot
# =====================================================================

feature_names = X.columns
importances = model.feature_importances_
FI = pd.DataFrame({"Feature": feature_names, "Importance": importances})
FI = FI.sort_values("Importance", ascending=False)

# Plot top-10 features
top10 = FI.head(10).iloc[::-1]

plt.figure(figsize=(8, 6))
plt.barh(top10["Feature"], top10["Importance"])
plt.xlabel("Importance Score")
plt.title("Top-10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance_top10.png")
plt.close()

# =====================================================================
# 12) Calibration Curve
# =====================================================================

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=50)

plt.figure(figsize=(7, 7))
plt.scatter(prob_pred, prob_true, edgecolor="k")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Probability")
plt.title("Calibration Curve")
plt.tight_layout()
plt.savefig("calibration_curve.png")
plt.close()