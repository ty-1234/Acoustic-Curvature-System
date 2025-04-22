import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# === Load Dataset ===
DATASET_PATH = "csv_data/combined_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# === Select FFT columns only ===
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]

# === Classification: Predict Section ===
df_class = df.dropna(subset=["Section"])
y_class = df_class["Section"].astype(str)
X_class = df_class[fft_cols]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, stratify=y_class, test_size=0.2, random_state=42)

clf = LazyClassifier(verbose=1, ignore_warnings=True)
models_class, _ = clf.fit(Xc_train, Xc_test, yc_train, yc_test)

# === Regression: Predict Curvature ===
df_reg = df.dropna(subset=["Curvature_Label"])
y_reg = df_reg["Curvature_Label"]
X_reg = df_reg[fft_cols]

# Normalize FFT features for regression
scaler = MinMaxScaler()
Xr_scaled = scaler.fit_transform(X_reg)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled, y_reg, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=1, ignore_warnings=True)
models_reg, _ = reg.fit(Xr_train, Xr_test, yr_train, yr_test)

# === Save Results ===
os.makedirs("results", exist_ok=True)
models_class.to_csv("results/section_classification_results.csv")
models_reg.to_csv("results/curvature_regression_results.csv")

print("✅ Classification and regression benchmarking complete.")
print("📁 Results saved to: results/section_classification_results.csv")
print("📁 Results saved to: results/curvature_regression_results.csv")
