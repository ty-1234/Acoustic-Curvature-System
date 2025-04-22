import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from tqdm import tqdm

# === Load Dataset ===
DATASET_PATH = "csv_data/combined_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# === Select FFT columns only ===
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]

# === Classification Task ===
print("\n🎯 Running LazyClassifier (Section Classification)...")
df_class = df.dropna(subset=["Section"])
y_class = df_class["Section"].astype(str)
X_class = df_class[fft_cols]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, stratify=y_class, test_size=0.2, random_state=42
)

start = time.time()
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models_class, _ = clf.fit(Xc_train, Xc_test, yc_train, yc_test)
duration_class = time.time() - start

# === Show Top 5 Classifiers ===
print(f"\n🏁 Finished LazyClassifier in {duration_class:.2f} seconds.")
print("\n🏆 Top 5 Classifier Models (by Accuracy):")
print(models_class.sort_values("Accuracy", ascending=False).head(5))

# === Regression Task ===
print("\n📐 Running LazyRegressor (Curvature Estimation)...")
df_reg = df.dropna(subset=["Curvature_Label"])
y_reg = df_reg["Curvature_Label"]
X_reg = df_reg[fft_cols]

scaler = MinMaxScaler()
Xr_scaled = scaler.fit_transform(X_reg)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr_scaled, y_reg, test_size=0.2, random_state=42
)

start = time.time()
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_reg, _ = reg.fit(Xr_train, Xr_test, yr_train, yr_test)
duration_reg = time.time() - start

# === Show Top 5 Regressors ===
print(f"\n🏁 Finished LazyRegressor in {duration_reg:.2f} seconds.")
print("\n🏆 Top 5 Regressor Models (by RMSE):")
print(models_reg.sort_values("RMSE").head(5))

# === Save to CSV ===
os.makedirs("results", exist_ok=True)
models_class.to_csv("results/section_classification_results.csv")
models_reg.to_csv("results/curvature_regression_results.csv")

print("\n✅ Results saved to:")
print("   - results/section_classification_results.csv")
print("   - results/curvature_regression_results.csv")
