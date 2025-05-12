import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models", "extratrees")
test_data_path = os.path.join(script_dir, "data")  # UPDATE THIS

# === Load model + scaler ===
model = joblib.load(os.path.join(model_dir, "extratrees_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "extratrees_scaler.pkl"))

# === Define feature set used ===
features_used = [
    "FFT_1400Hz", "FFT_800Hz", "FFT_1800Hz",
    "High_Band_Mean", "Mid_Band_Mean",
    "Norm_FFT_1400Hz", "Norm_FFT_800Hz",
    "Norm_High_Band_Mean", "Position_cm"
]

# === Load and prepare test data ===
predictions = []
for fname in os.listdir(test_data_path):
    if not fname.endswith(".csv"):
        continue

    file_path = os.path.join(test_data_path, fname)
    df = pd.read_csv(file_path)

    if "Curvature_Active" in df.columns:
        df = df[df["Curvature_Active"] == 1]

    if df.empty:
        continue

    X = df[features_used]
    y_true = df["Curvature"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n📄 {fname} → R² = {r2:.4f}, RMSE = {rmse:.6f}")

    df["Predicted_Curvature"] = y_pred
    output_csv = os.path.join(model_dir, f"predictions_{fname}")
    df.to_csv(output_csv, index=False)
    predictions.append((fname, r2, rmse))

    # Save scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True Curvature")
    plt.ylabel("Predicted Curvature")
    plt.title(f"{fname}\nR² = {r2:.3f}, RMSE = {rmse:.6f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"plot_{fname.replace('.csv', '')}.png"))
    plt.close()
    
    # Save residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel("Predicted Curvature")
    plt.ylabel("Residual")
    plt.title(f"{fname} - Residuals\nMean Error: {np.mean(residuals):.6f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"residuals_{fname.replace('.csv', '')}.png"))
    plt.close()

# Calculate overall metrics if multiple files were processed
if len(predictions) > 1:
    avg_r2 = np.mean([r2 for _, r2, _ in predictions])
    avg_rmse = np.mean([rmse for _, _, rmse in predictions])
    print(f"\n📊 Overall: Average R² = {avg_r2:.4f}, Average RMSE = {avg_rmse:.6f}")

# Final summary
print("\n=== Evaluation Summary ===")
for fname, r2, rmse in predictions:
    print(f"{fname}: R² = {r2:.4f}, RMSE = {rmse:.6f}")

# Export summary to CSV
summary_df = pd.DataFrame(predictions, columns=["File", "R2", "RMSE"])
summary_df.to_csv(os.path.join(model_dir, "evaluation_summary.csv"), index=False)
print(f"\n✅ Saved evaluation summary to {os.path.join(model_dir, 'evaluation_summary.csv')}")