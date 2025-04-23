import pandas as pd
import os
import joblib
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load data ===
df = pd.read_csv("csv_data/combined_dataset.csv")
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]
y_class = df["Section"].astype(str)
y_reg = df["Curvature_Label"]
groups = df["RunID"]

# === Scale features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Save scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/fft_scaler.pkl")

# === Choose models ===
classifier_models = {
    "extratrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "lgbm": LGBMClassifier(random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
}

regressor_models = {
    "xgb": XGBRegressor(n_estimators=100, random_state=42),
    "lgbm": LGBMRegressor(n_estimators=100, random_state=42),
    "histgb": HistGradientBoostingRegressor(random_state=42),
}

# === Pick which ones to train ===
clf_name = "extratrees"  # change this to 'lgbm' or 'rf' if needed
reg_name = "xgb"         # change this to 'lgbm' or 'histgb'

clf = classifier_models[clf_name]
reg = regressor_models[reg_name]

# === Split for evaluation ===
gkf = GroupKFold(n_splits=5)

# === Classification Evaluation ===
acc_scores = cross_val_score(clf, X_scaled, y_class, cv=gkf.split(X_scaled, y_class, groups=groups), scoring="accuracy")
print(f"\n✅ Classifier ({clf_name}) Accuracy (GroupKFold): {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

# Train final classifier on full data and save
clf.fit(X_scaled, y_class)
joblib.dump(clf, f"models/{clf_name}_classifier.pkl")

# Confusion Matrix Plot
y_class_pred = clf.predict(X_scaled)
cm = confusion_matrix(y_class, y_class_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title(f"Confusion Matrix: {clf_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"models/confusion_{clf_name}.png")
plt.clf()

# === Regression Evaluation ===
rmse_scores = []
r2_scores = []

for train_idx, test_idx in gkf.split(X_scaled, y_reg, groups):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))
    r2_scores.append(r2_score(y_test, y_pred))

print(f"\n📐 Regressor ({reg_name}) RMSE (GroupKFold): {np.mean(rmse_scores):.5f} ± {np.std(rmse_scores):.5f}")
print(f"📊 Regressor ({reg_name}) R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Train final regressor on full data and save
reg.fit(X_scaled, y_reg)
joblib.dump(reg, f"models/{reg_name}_regressor.pkl")

# True vs Predicted Plot
y_pred_full = reg.predict(X_scaled)
plt.scatter(y_reg, y_pred_full, alpha=0.3)
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title(f"True vs Predicted: {reg_name}")
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--')
plt.tight_layout()
plt.savefig(f"models/true_vs_predicted_{reg_name}.png")
plt.clf()
