#1 Load and Prepare Data
import pandas as pd

# Load the Rifampicin dataset
data = pd.read_csv("rifampicin.csv")

# Separate features and label
X = data.drop(columns=["Isolate", "rifampicin"])
y = data["rifampicin"]

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

# Optional: check balance of resistant vs susceptible isolates
print(y.value_counts())

#2 Split into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#3 Train the Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,          # number of trees
    max_depth=None,            # trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)

# Train the model
rf_model.fit(X_train, y_train)

#4 Make Predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # probability of being resistant

#5 Evaluate Model Performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, y_prob))
print("AUPR:", average_precision_score(y_test, y_prob))

#6 Visualize Model Results
#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Rifampicin Resistance (Random Forest)")
plt.show()

#Feature Importance (Top Predictive SNPs)
import numpy as np

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,5))
plt.title("Top 20 Important SNPs - Random Forest")
plt.bar(range(20), importances[indices[:20]])
plt.xticks(range(20), X.columns[indices[:20]], rotation=90)
plt.tight_layout()
plt.show()

#7 SHAP Analysis for Explainability
import shap
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("SHAP ANALYSIS FOR RANDOM FOREST - RIFAMPICIN RESISTANCE")
print("="*60)

# Create explainer
print("\n>>> Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values_raw = explainer.shap_values(X_test)

# Handle binary classification output - extract class 1 (resistant)
if isinstance(shap_values_raw, list):
    shap_vals = np.array(shap_values_raw[1], dtype=np.float64)
elif len(shap_values_raw.shape) == 3:
    shap_vals = np.array(shap_values_raw[:, :, 1], dtype=np.float64)
else:
    shap_vals = np.array(shap_values_raw, dtype=np.float64)

# Convert X_test to float64 numpy array
X_test_array = np.array(X_test.values, dtype=np.float64)

print(f"SHAP values shape: {shap_vals.shape}")
print(f"X_test shape: {X_test_array.shape}")

# Calculate mean absolute SHAP for feature importance
mean_abs_shap = np.abs(shap_vals).mean(axis=0)

# Get feature names
feature_names = X_test.columns.tolist()

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_abs_shap
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string())

# ============ SHAP BEESWARM PLOT ============
print("\n>>> Generating SHAP Beeswarm Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_test_array,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.xlabel('SHAP value (impact on model output)', fontsize=11)
plt.title('SHAP Values of SNPs in the Random Forest Model', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm_rf.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: shap_beeswarm_rf.png")

# ============ SHAP BAR PLOT (Mean Absolute SHAP) ============
print("\n>>> Generating SHAP Bar Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_test_array,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.xlabel('Mean(|SHAP value|)', fontsize=11)
plt.title('Feature Importance - Random Forest', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar_rf.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: shap_bar_rf.png")

# Save importance rankings
importance_df.to_csv('shap_feature_importance_rf.csv', index=False)
print("\n✓ Saved feature importance to: shap_feature_importance_rf.csv")

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE!")
print("="*60)