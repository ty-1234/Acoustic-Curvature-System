# gpr_eval.py
# ------------------------------------------------------------
import warnings, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

# --------- paths ------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PKL  = BASE_DIR / "models" / "gpr" / "gpr_ard.pkl"
TEST_DIR   = BASE_DIR / "data"                    # put hold‑out CSVs here
PLOT_EACH  = True                                 # set False to skip PNGs

# --------- load model bundle ------------------------------------------------
bundle      = joblib.load(MODEL_PKL)
gpr         = bundle["gpr"]
scaler      = bundle["scaler"]
feat_cols   = bundle["features"]

print(f"Loaded GPR with {len(feat_cols)} features:\n{feat_cols}\n")

# --------- evaluate ---------------------------------------------------------
global_X, global_y = [], []
file_results = []

for csv_path in sorted(TEST_DIR.glob("*.csv")):
    df = pd.read_csv(csv_path)

    # keep only active rows (if column exists)
    if "Curvature_Active" in df.columns:
        df = df[df.Curvature_Active == 1]

    if df.empty:
        warnings.warn(f"{csv_path.name} has no active rows – skipped.")
        continue

    # check feature presence
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        warnings.warn(f"{csv_path.name} missing {missing} – skipped.")
        continue

    X = df[feat_cols].fillna(0).values
    y = df["Curvature"].values
    y_hat = gpr.predict(scaler.transform(X))

    rmse = np.sqrt(mean_squared_error(y, y_hat))

    # R² is undefined if y is constant
    if np.unique(y).size < 2:
        r2_disp = "—"
        warnings.warn(f"{csv_path.name} only one curvature – R² not reported.")
    else:
        r2_disp = f"{r2_score(y, y_hat):.4f}"

    print(f"{csv_path.name:35s}  RMSE = {rmse:.6f}  R² = {r2_disp}")
    file_results.append((csv_path.name, r2_disp, rmse))

    # optional scatter plot --------------------------------------------------
    if PLOT_EACH:
        plt.figure(figsize=(4,4))
        plt.scatter(y, y_hat, alpha=.6, edgecolors="k")
        lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
        plt.plot(lims, lims, "r--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(csv_path.stem)
        plt.tight_layout()
        plt.savefig(MODEL_PKL.parent / f"plot_{csv_path.stem}.png", dpi=160)
        plt.close()

    # collect for global score
    global_X.append(X)
    global_y.append(y)

# --------- global metrics ---------------------------------------------------
if global_X:
    X_all = np.vstack(global_X)
    y_all = np.concatenate(global_y)
    y_all_hat = gpr.predict(scaler.transform(X_all))

    print("\n=== GLOBAL METRICS over all test rows ===")
    print("RMSE :", np.sqrt(mean_squared_error(y_all, y_all_hat)))
    print("R²   :", r2_score(y_all, y_all_hat))

# --------- per‑file summary -------------------------------------------------
print("\n=== PER‑FILE SUMMARY ===")
for name, r2, rmse in file_results:
    print(f"{name:35s}  RMSE = {rmse:.6f}  R² = {r2}")