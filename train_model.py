
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create output directory
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv('data/creditcard.csv')

# Print columns for debugging
print("Columns in dataset:", data.columns.tolist())

# Use only numeric columns for features
target_col = 'is_fraud'
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
X = data[numeric_cols]
y = data[target_col]


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X_pca)

# Predict anomalies
y_pred = model.predict(X_pca)
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Evaluation
print("Classification Report (Isolation Forest):")
print(classification_report(y, y_pred_binary))
print("ROC AUC Score:", roc_auc_score(y, y_pred_binary))

# Confusion Matrix
cm = confusion_matrix(y, y_pred_binary)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model, scaler, and PCA
joblib.dump(model, 'models/isolation_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca_model.pkl')
