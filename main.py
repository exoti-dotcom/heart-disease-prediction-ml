# ==========================================
# HEART DISEASE PREDICTION USING MACHINE LEARNING
# ==========================================

# ---------- Import Libraries ----------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score
)

# ==========================================
# PROJECT PATH SETUP
# ==========================================

# Gets the exact folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

# Output folder path
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create outputs folder if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "heart.csv not found. Please place it inside the data folder:\n"
        f"{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# ---------- Display Dataset ----------
print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================================
# FEATURE AND TARGET SPLIT
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# MODEL 1: LOGISTIC REGRESSION
# ==========================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# MODEL 2: DECISION TREE
# ==========================================

tree_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    max_depth=4
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# ==========================================
# MODEL 3: RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ==========================================
# FUNCTION FOR MODEL EVALUATION
# ==========================================

def evaluate_model(name, y_test, y_pred, filename):
    print(f"\n========== {name} ==========\n")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save confusion matrix image inside outputs folder
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix: {save_path}")

# ==========================================
# EVALUATE ALL MODELS
# ==========================================

evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_log,
    "logistic_regression_confusion_matrix.png"
)

evaluate_model(
    "Decision Tree",
    y_test,
    y_pred_tree,
    "decision_tree_confusion_matrix.png"
)

evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf,
    "random_forest_confusion_matrix.png"
)

# ==========================================
# ROC CURVE COMPARISON
# ==========================================

# Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = roc_auc_score(y_test, y_prob_log)

# Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_tree = roc_auc_score(y_test, y_prob_tree)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

# ---------- Plot ROC Curves ----------
plt.figure(figsize=(8, 6))

plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {auc_log:.2f})"
)

plt.plot(
    fpr_tree,
    tpr_tree,
    label=f"Decision Tree (AUC = {auc_tree:.2f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.2f})"
)

# Random Guess Line
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

# Save ROC curve image inside outputs folder
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.close()

print(f"\nSaved ROC curve: {roc_path}")

# ==========================================
# FINAL MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_tree),
        accuracy_score(y_test, y_pred_rf)
    ],

    "Precision": [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_tree),
        precision_score(y_test, y_pred_rf)
    ],

    "Recall": [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_tree),
        recall_score(y_test, y_pred_rf)
    ],

    "AUC": [
        auc_log,
        auc_tree,
        auc_rf
    ]
})

print("\n========== FINAL MODEL COMPARISON ==========\n")
print(results)

# Save final comparison table as CSV
results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
results.to_csv(results_path, index=False)

print(f"\nSaved model comparison results: {results_path}")

# ==========================================
# BEST MODEL SELECTION
# ==========================================

best_model = results.sort_values(
    by="Recall",
    ascending=False
)

print("\nBest Model Based on Recall:\n")
print(best_model.head(1))

print("\nAll output graphs and results have been saved successfully.")
print(f"Output folder location: {OUTPUT_DIR}")