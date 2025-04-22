import pandas as pd
import os
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from tqdm import tqdm

# === Add parent directory to path for imports ===
# This allows us to access the csv_data directory from the neural_network subdirectory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# === Set correct paths ===
DATASET_PATH = os.path.join(parent_dir, "csv_data", "combined_dataset.csv")
RESULTS_DIR = os.path.join(parent_dir, "csv_data")

print(f"🔍 Loading dataset from: {DATASET_PATH}")

# === Load Dataset ===
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
os.makedirs(RESULTS_DIR, exist_ok=True)
class_results_path = os.path.join(RESULTS_DIR, "section_classification_results.csv")
reg_results_path = os.path.join(RESULTS_DIR, "curvature_regression_results.csv")

models_class.to_csv(class_results_path)
models_reg.to_csv(reg_results_path)

print("\n✅ Results saved to:")
print(f"   - {class_results_path}")
print(f"   - {reg_results_path}")
